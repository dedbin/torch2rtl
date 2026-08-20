from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import subprocess
import sys
import tomllib
import types
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

import torch2rtl.cli as cli
import torch2rtl.frontend.pytorch_fx as fx_frontend
from torch2rtl.backend.systemverilog.emit import emit_systemverilog
from torch2rtl.frontend.pytorch_fx import UnsupportedOpError, parse_model
from torch2rtl.ir.ops import ArgmaxIR, Conv2dIR, ReluIR
from torch2rtl.quant.fixed_point import FixedPointConfig, quantize_array
from torch2rtl.quant.reference import (
    QuantizedConv2dIR,
    QuantizedGraph,
    QuantizedLinearIR,
    conv2d_fixed,
    infer_float_graph,
    infer_quantized,
    linear_fixed,
    quantize_graph,
    validate_accumulator_width,
)
from torch2rtl.synth.report import update_report
from torch2rtl.synth.yosys import SynthResult, run_yosys
from torch2rtl.verify.simulator import SimulationResult, _simulation_ok, run_simulation


_TRACE_GLOBAL_SWITCH = 0
_HELPER_TRACE_SWITCH = 0
_DESCRIPTOR_TRACE_SWITCH = 0
_SPOOFED_MODULE_TRACE_SWITCH = 0
_ARRAY_STATE_TRACE_SWITCH = 0
_INT_STATE_TRACE_SWITCH = 0
_STATIC_DESCRIPTOR_TRACE_SWITCH = 0
_NESTED_CODE_TRACE_SWITCH = 0


def _bump_nested_code_trace_switch() -> int:
    global _NESTED_CODE_TRACE_SWITCH
    _NESTED_CODE_TRACE_SWITCH += 1
    return _NESTED_CODE_TRACE_SWITCH



def test_explicit_verify_and_synth_fail_when_tool_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        cli,
        "run_simulation",
        lambda _path: SimulationResult(False, "not_found", "simulator not found"),
    )
    monkeypatch.setattr(
        cli,
        "run_yosys",
        lambda _path: SynthResult(False, "not_found", "yosys not found"),
    )

    assert cli.cmd_verify(argparse.Namespace(build_dir=tmp_path)) == 1
    assert cli.cmd_synth(argparse.Namespace(build_dir=tmp_path)) == 1


def test_pytest_configuration_does_not_require_repo_local_temp_directory() -> None:
    config = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "--basetemp=temp/pytest" not in config


def test_tiny_mlp_real_train_checkpoint_compile_path(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny_mlp.pt"
    out_dir = tmp_path / "rtl"
    train = subprocess.run(
        [
            sys.executable,
            "examples/tiny_mlp/train.py",
            "--epochs",
            "1",
            "--out",
            str(checkpoint),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert train.returncode == 0, train.stdout + train.stderr
    compile_result = subprocess.run(
        [
            sys.executable,
            "examples/tiny_mlp/compile.py",
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(out_dir),
            "--vectors",
            "2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr
    assert checkpoint.exists()
    assert (out_dir / "top.sv").exists()
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report["vectors"]["count"] == 2
    assert str(checkpoint) in compile_result.stdout

    trained_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    trained_weight = trained_state["net.0.weight"].detach().numpy()
    torch.manual_seed(0)
    from examples.tiny_mlp.model import create_model

    initial_weight = create_model().net[0].weight.detach().numpy()
    cfg = FixedPointConfig()
    trained_q = quantize_array(trained_weight, cfg).reshape(-1)
    initial_q = quantize_array(initial_weight, cfg).reshape(-1)
    changed = np.flatnonzero(trained_q != initial_q)
    assert changed.size > 0
    weight_index = int(changed[0])
    trained_value = int(trained_q[weight_index])
    literal = (
        f"-{cfg.bits}'sd{abs(trained_value)}"
        if trained_value < 0
        else f"{cfg.bits}'sd{trained_value}"
    )
    top = (out_dir / "top.sv").read_text(encoding="utf-8")
    assert (
        f"NET_0_0_WEIGHTS[{weight_index}*DATA_BITS +: DATA_BITS] = {literal};"
        in top
    )


def test_wheel_configuration_includes_runtime_demo_models() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    force_include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"][
        "force-include"
    ]
    required = {
        "examples/tiny_mlp/model.py",
        "examples/grid_classifier/model.py",
        "examples/tiny_conv/model.py",
        "examples/image_cnn/model.py",
        "examples/image_cnn/data.py",
    }

    assert required <= set(force_include)
    assert all(force_include[path] == path for path in required)


def test_v02_python_boundary_is_synchronized_across_release_sources() -> None:
    required_python = ">=3.12,<3.13"
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads(Path("uv.lock").read_text(encoding="utf-8"))

    assert pyproject["project"]["requires-python"] == required_python
    assert lock["requires-python"] == required_python

    release_docs = {
        "README.md": "Python `>=3.12,<3.13`",
        "docs/v0.2_semantic_correctness.md": "Python `>=3.12,<3.13`",
        "docs/torch2rtl_guide.md": "Python `>=3.12,<3.13`",
    }
    for filename, required_text in release_docs.items():
        text = Path(filename).read_text(encoding="utf-8")
        assert required_text in text, filename
        assert "Python 3.11+" not in text, filename
        assert "Python `3.11+`" not in text, filename
        assert "Python 3.12+" not in text, filename
        assert "Python `3.12+`" not in text, filename

    readme = Path("README.md").read_text(encoding="utf-8")
    assert "badge/Python-3.12-" in readme
    assert "badge/Python-3.11" not in readme


def test_v02_integrity_documentation_states_local_manifest_threat_boundary() -> None:
    for filename in (
        "README.md",
        "docs/v0.2_semantic_correctness.md",
        "docs/torch2rtl_guide.md",
    ):
        text = Path(filename).read_text(encoding="utf-8")
        normalized = " ".join(text.split())
        assert "не является внешним корнем доверия" in normalized, filename
        assert "согласован" in text and "вне" in text, filename
        assert "threat model" in text or "модели защиты" in text, filename
        assert "внешн" in normalized and "подпис" in normalized, filename
        assert "доверенн" in normalized and "manifest" in normalized, filename
        assert "не гарант" in normalized, filename
        assert "произволь" in normalized and "vector payload" in normalized, filename
