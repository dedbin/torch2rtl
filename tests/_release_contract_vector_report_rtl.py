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



def test_emit_rejects_zero_vectors_before_writing_artifacts(tmp_path: Path) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))

    with pytest.raises(ValueError, match="vector_count"):
        emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=0)

    assert not tmp_path.exists() or not tuple(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("filename", "mutation"),
    [
        ("input_vectors.txt", "empty"),
        ("expected_classes.txt", "empty"),
        ("expected_logits.txt", "empty"),
        ("input_vectors.txt", "short"),
        ("expected_classes.txt", "short"),
        ("expected_logits.txt", "short"),
        ("input_vectors.txt", "extra"),
        ("expected_classes.txt", "extra"),
        ("expected_logits.txt", "extra"),
    ],
)
def test_simulation_rejects_malformed_vector_file_lengths(
    tmp_path: Path,
    filename: str,
    mutation: str,
) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    path = tmp_path / filename
    tokens = path.read_text(encoding="utf-8").split()
    if mutation == "empty":
        tokens = []
    elif mutation == "short":
        tokens = tokens[:-1]
    else:
        tokens.append("0")
    path.write_text("\n".join(tokens) + ("\n" if tokens else ""), encoding="utf-8")

    result = run_simulation(tmp_path)

    assert not result.ok
    assert result.status == "invalid_vectors" or "FAIL" in result.stdout


@pytest.mark.parametrize(
    ("filename", "delta"),
    [
        ("input_vectors.txt", 1 << 8),
        ("expected_classes.txt", 2),
        ("expected_logits.txt", 1 << 32),
        ("input_vectors.txt", 1 << 64),
        ("expected_classes.txt", 1 << 64),
        ("expected_logits.txt", 1 << 64),
        ("input_vectors.txt", "x"),
        ("expected_classes.txt", "x"),
        ("expected_logits.txt", "x"),
        ("input_vectors.txt", "z"),
        ("expected_classes.txt", "z"),
        ("expected_logits.txt", "z"),
    ],
)
def test_simulation_rejects_vector_values_that_alias_after_sv_truncation(
    tmp_path: Path,
    filename: str,
    delta: int | str,
) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    path = tmp_path / filename
    tokens = path.read_text().split()
    if isinstance(delta, str):
        tokens[-1] = delta
    else:
        tokens = [str(int(token) + delta) for token in tokens]
    path.write_text("\n".join(tokens) + "\n", encoding="utf-8")

    result = run_simulation(tmp_path)

    assert not result.ok
    assert result.status == "invalid_vectors"
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["simulation"]["status"] == "invalid_vectors"


@pytest.mark.parametrize(
    "filename",
    ["input_vectors.txt", "expected_classes.txt", "expected_logits.txt"],
)
def test_simulation_rejects_noncanonical_decimal_vector_tokens(
    tmp_path: Path,
    filename: str,
) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    path = tmp_path / filename
    tokens = path.read_text(encoding="utf-8").split()
    tokens[0] = "+0"
    path.write_text("\n".join(tokens) + "\n", encoding="utf-8")

    result = run_simulation(tmp_path)

    assert not result.ok
    assert result.status == "invalid_vectors"
    assert "canonical signed decimal" in result.message


@pytest.mark.parametrize(
    ("filename", "delta"),
    [
        ("input_vectors.txt", 1 << 8),
        ("expected_classes.txt", 2),
        ("expected_logits.txt", 1 << 32),
        ("input_vectors.txt", 1 << 64),
        ("expected_classes.txt", 1 << 64),
        ("expected_logits.txt", 1 << 64),
        ("input_vectors.txt", "x"),
        ("expected_classes.txt", "x"),
        ("expected_logits.txt", "x"),
        ("input_vectors.txt", "z"),
        ("expected_classes.txt", "z"),
        ("expected_logits.txt", "z"),
    ],
)
def test_generated_testbench_itself_rejects_values_outside_port_ranges(
    tmp_path: Path,
    filename: str,
    delta: int | str,
) -> None:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog is None or vvp is None:
        pytest.skip("Icarus Verilog is required for direct testbench regression")
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    path = tmp_path / filename
    tokens = path.read_text().split()
    if isinstance(delta, str):
        tokens[-1] = delta
    else:
        tokens = [str(int(token) + delta) for token in tokens]
    path.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    sources = [
        "linear_comb.sv",
        "relu.sv",
        "argmax.sv",
        "top.sv",
        "tb_top.sv",
    ]
    compiled = subprocess.run(
        [iverilog, "-g2012", "-o", "simv-direct", *sources],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr

    simulated = subprocess.run(
        [vvp, "-M", "-", "simv-direct"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert simulated.returncode != 0
    assert "FAIL" in simulated.stdout


def test_vector_contract_parser_ignores_commented_localparam_alias(
    tmp_path: Path,
) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    inputs = tmp_path / "input_vectors.txt"
    tokens = [str(int(token) + 256) for token in inputs.read_text().split()]
    inputs.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    metadata_path = tmp_path / "vectors.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["data_bits"] = 16
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    testbench = tmp_path / "tb_top.sv"
    testbench.write_text(
        "// localparam int DATA_BITS = 16;\n" + testbench.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "visualization.json").unlink()

    result = run_simulation(tmp_path)

    assert not result.ok
    assert result.status == "invalid_vectors"


def test_vector_contract_parser_ignores_strings_that_look_like_comments_or_params(
    tmp_path: Path,
) -> None:
    model = nn.Sequential(nn.Linear(2, 1)).eval()
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.zero_()
    graph = parse_model(model, input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    (tmp_path / "input_vectors.txt").write_text("0 0\n0 0\n", encoding="utf-8")
    (tmp_path / "expected_classes.txt").write_text("0\n0\n", encoding="utf-8")
    (tmp_path / "expected_logits.txt").write_text("0\n0\n", encoding="utf-8")
    testbench = tmp_path / "tb_top.sv"
    text = testbench.read_text(encoding="utf-8")
    text = text.replace(
        "localparam int DATA_BITS = 8;",
        '\n'.join(
            [
                'initial $display("/*");',
                "localparam int DATA_BITS = 4;",
                'initial $display("*/");',
                'initial $display("localparam int DATA_BITS = 8;");',
            ]
        ),
        1,
    )
    testbench.write_text(text, encoding="utf-8")

    result = run_simulation(tmp_path)

    assert not result.ok
    assert result.status == "invalid_vectors"
    assert "header" in result.message


def test_vector_contract_rejects_systemverilog_preprocessor_directives(
    tmp_path: Path,
) -> None:
    model = nn.Sequential(nn.Linear(2, 1)).eval()
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.zero_()
    graph = parse_model(model, input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    (tmp_path / "input_vectors.txt").write_text("0 0\n0 0\n", encoding="utf-8")
    (tmp_path / "expected_classes.txt").write_text("0\n0\n", encoding="utf-8")
    (tmp_path / "expected_logits.txt").write_text("0\n0\n", encoding="utf-8")
    testbench = tmp_path / "tb_top.sv"
    text = testbench.read_text(encoding="utf-8")
    text = text.replace(
        "localparam int DATA_BITS = 8;",
        "\n".join(
            [
                "`define TORCH2RTL_LOCALPARAM localparam",
                "`define TORCH2RTL_DATA_BITS int DATA_BITS = 4",
                "`ifdef TORCH2RTL_INACTIVE",
                "localparam int DATA_BITS = 8;",
                "`else",
                "`TORCH2RTL_LOCALPARAM `TORCH2RTL_DATA_BITS;",
                "`endif",
            ]
        ),
        1,
    )
    testbench.write_text(text, encoding="utf-8")

    result = run_simulation(tmp_path)

    assert not result.ok
    assert result.status == "invalid_vectors"
    assert "preprocessor" in result.message


def test_vector_contract_requires_generated_top_level_parameter_header(
    tmp_path: Path,
) -> None:
    model = nn.Sequential(nn.Linear(2, 1)).eval()
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.zero_()
    graph = parse_model(model, input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    (tmp_path / "input_vectors.txt").write_text("0 0\n0 0\n", encoding="utf-8")
    (tmp_path / "expected_classes.txt").write_text("0\n0\n", encoding="utf-8")
    (tmp_path / "expected_logits.txt").write_text("0\n0\n", encoding="utf-8")
    testbench = tmp_path / "tb_top.sv"
    text = testbench.read_text(encoding="utf-8")
    text = text.replace(
        "localparam int DATA_BITS = 8;",
        "\n".join(
            [
                "parameter int DATA_BITS = 4;",
                "generate",
                "    if (0) begin : inactive_contract",
                "        localparam int DATA_BITS = 8;",
                "    end",
                "endgenerate",
            ]
        ),
        1,
    )
    testbench.write_text(text, encoding="utf-8")

    result = run_simulation(tmp_path)

    assert not result.ok
    assert result.status == "invalid_vectors"
    assert "header" in result.message


def test_vector_contract_rejects_modified_generated_checker_body(
    tmp_path: Path,
) -> None:
    model = nn.Sequential(nn.Linear(2, 1)).eval()
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.zero_()
    graph = parse_model(model, input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    (tmp_path / "input_vectors.txt").write_text("0 0\n0 0\n", encoding="utf-8")
    (tmp_path / "expected_classes.txt").write_text("0\n0\n", encoding="utf-8")
    (tmp_path / "expected_logits.txt").write_text("0\n0\n", encoding="utf-8")
    testbench = tmp_path / "tb_top.sv"
    text = testbench.read_text(encoding="utf-8")
    text = text.replace(
        "endmodule",
        'initial begin $display("PASS vectors=2"); $finish(0); end\nendmodule',
        1,
    )
    testbench.write_text(text, encoding="utf-8")

    result = run_simulation(tmp_path)

    assert not result.ok
    assert result.status == "invalid_vectors"
    assert "checker body" in result.message


def test_vector_contract_rejects_simulation_control_in_dut_source(
    tmp_path: Path,
) -> None:
    model = nn.Sequential(nn.Linear(2, 1)).eval()
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.zero_()
    graph = parse_model(model, input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    (tmp_path / "input_vectors.txt").write_text("0 0\n0 0\n", encoding="utf-8")
    (tmp_path / "expected_classes.txt").write_text("0\n0\n", encoding="utf-8")
    (tmp_path / "expected_logits.txt").write_text("0\n0\n", encoding="utf-8")
    top = tmp_path / "top.sv"
    text = top.read_text(encoding="utf-8")
    top.write_text(
        text.replace(
            "endmodule",
            'initial begin $display("PASS vectors=2"); $finish(0); end\nendmodule',
            1,
        ),
        encoding="utf-8",
    )

    result = run_simulation(tmp_path)

    assert not result.ok
    assert result.status == "invalid_vectors"
    assert "DUT source" in result.message


@pytest.mark.parametrize(
    "injection",
    [
        "defparam DATA_BITS = 16;",
        "wire checker_probe; assign checker_probe = tb_top.status;",
    ],
)
def test_vector_contract_binds_every_rtl_source_to_compile_report(
    tmp_path: Path,
    injection: str,
) -> None:
    model = nn.Sequential(nn.Linear(1, 1)).eval()
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.zero_()
    graph = parse_model(model, input_shape=(1,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=1, seed=4)
    (tmp_path / "input_vectors.txt").write_text("0\n", encoding="utf-8")
    (tmp_path / "expected_classes.txt").write_text("0\n", encoding="utf-8")
    (tmp_path / "expected_logits.txt").write_text("0\n", encoding="utf-8")
    top = tmp_path / "top.sv"
    top.write_text(
        top.read_text(encoding="utf-8").replace(
            "endmodule",
            f"{injection}\nendmodule",
            1,
        ),
        encoding="utf-8",
    )

    result = run_simulation(tmp_path)

    assert not result.ok
    assert result.status == "invalid_vectors"
    assert "integrity" in result.message
    synthesis = run_yosys(tmp_path)
    assert not synthesis.ok
    assert synthesis.status == "invalid_sources"
    assert "integrity" in synthesis.message


def test_vector_contract_binds_testbench_width_to_compile_report(
    tmp_path: Path,
) -> None:
    model = nn.Sequential(nn.Linear(2, 1)).eval()
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.zero_()
    graph = parse_model(model, input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    (tmp_path / "input_vectors.txt").write_text("0 0\n0 0\n", encoding="utf-8")
    (tmp_path / "expected_classes.txt").write_text("0\n0\n", encoding="utf-8")
    (tmp_path / "expected_logits.txt").write_text("0\n0\n", encoding="utf-8")
    metadata_path = tmp_path / "vectors.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["data_bits"] = 16
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    testbench = tmp_path / "tb_top.sv"
    text = testbench.read_text(encoding="utf-8")
    text = text.replace("DATA_BITS = 8", "DATA_BITS = 16", 1)
    text = text.replace("DATA_MIN = -64'sd128", "DATA_MIN = -64'sd32768", 1)
    text = text.replace("DATA_MAX = 64'sd127", "DATA_MAX = 64'sd32767", 1)
    testbench.write_text(text, encoding="utf-8")

    result = run_simulation(tmp_path)

    assert not result.ok
    assert result.status == "invalid_vectors"
    assert "compile report" in result.message


def test_synthesis_result_does_not_depend_on_valid_vector_trace(tmp_path: Path) -> None:
    if shutil.which("yosys") is None:
        pytest.skip("Yosys is required for synthesis regression")
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    (tmp_path / "input_vectors.txt").write_text("", encoding="utf-8")

    result = run_yosys(tmp_path)

    assert result.ok
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["synthesis"]["status"] == "passed"
    manifest = json.loads(
        (tmp_path / "visualization.json").read_text(encoding="utf-8")
    )
    assert manifest["build"]["status"]["synthesis"]["status"] == "passed"
    assert manifest["trace"]["matched"] is False


def test_synthesis_result_survives_structurally_corrupt_visualization(
    tmp_path: Path,
) -> None:
    if shutil.which("yosys") is None:
        pytest.skip("Yosys is required for synthesis regression")
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    manifest_path = tmp_path / "visualization.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["quant"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_yosys(tmp_path)

    assert result.ok
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["synthesis"]["status"] == "passed"


def test_synthesis_rejects_report_width_mismatch_before_yosys(tmp_path: Path) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    report_path = tmp_path / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["quant"]["bits"] = 16
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = run_yosys(tmp_path)

    assert not result.ok
    assert result.status == "invalid_sources"
    assert "compile contract" in result.message


def test_synthesis_rejects_noninteger_report_vector_count(tmp_path: Path) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    report_path = tmp_path / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["vectors"]["count"] = 2.0
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = run_yosys(tmp_path)

    assert not result.ok
    assert result.status == "invalid_sources"
    assert "compile contract" in result.message


_MALFORMED_REPORT_CASES = (
    "invalid-json",
    "array-root",
    "status-only-number",
    "status-number",
    "quant-array",
    "vectors-array",
    "quant-bool",
    "vectors-float",
    "generated-files-number",
    "generated-files-item-number",
    "missing-metrics",
)

_HUGE_JSON_INTEGER = "9" * 5000
_DEEPLY_NESTED_JSON_ARRAY = "[" * 10000 + "0" + "]" * 10000
_EXTREME_JSON_PAYLOADS = (
    pytest.param(_HUGE_JSON_INTEGER, id="huge-integer-root"),
    pytest.param(
        '{"status": ' + _HUGE_JSON_INTEGER + "}",
        id="huge-integer-field",
    ),
    pytest.param(_DEEPLY_NESTED_JSON_ARRAY, id="deep-array-root"),
)


def _write_malformed_report(report_path: Path, case: str) -> None:
    if case == "invalid-json":
        report_path.write_text("{not json", encoding="utf-8")
        return
    if case == "array-root":
        report_path.write_text("[]", encoding="utf-8")
        return
    if case == "status-only-number":
        report_path.write_text('{"status": 1}', encoding="utf-8")
        return

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if case == "status-number":
        report["status"] = 1
    elif case == "quant-array":
        report["quant"] = []
    elif case == "vectors-array":
        report["vectors"] = []
    elif case == "quant-bool":
        report["quant"]["bits"] = True
    elif case == "vectors-float":
        report["vectors"]["count"] = 2.0
    elif case == "generated-files-number":
        report["generated_files"] = 1
    elif case == "generated-files-item-number":
        report["generated_files"] = ["top.sv", 1]
    elif case == "missing-metrics":
        del report["metrics"]
    else:
        raise AssertionError(f"unknown malformed report case: {case}")
    report_path.write_text(json.dumps(report), encoding="utf-8")


@pytest.mark.parametrize("case", _MALFORMED_REPORT_CASES)
def test_verify_rejects_malformed_compile_report_before_eda_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    case: str,
) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    _write_malformed_report(tmp_path / "report.json", case)

    def unexpected_eda_lookup(_name: str) -> str | None:
        pytest.fail("verify reached EDA discovery after failed report preflight")

    monkeypatch.setattr("torch2rtl.verify.simulator.find_eda_tool", unexpected_eda_lookup)

    result = run_simulation(tmp_path)
    rc = cli.cmd_verify(argparse.Namespace(build_dir=tmp_path))
    output = capsys.readouterr().out

    assert not result.ok
    assert "invalid compile report" in result.message
    assert rc == 1
    assert "invalid compile report" in output
    assert "Traceback" not in output
    recorded = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert recorded["preflight_failure"]["section"] == "simulation"
    assert recorded["status"]["simulation"]["ok"] is False


@pytest.mark.parametrize("case", _MALFORMED_REPORT_CASES)
def test_synth_rejects_malformed_compile_report_before_eda_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    case: str,
) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    _write_malformed_report(tmp_path / "report.json", case)

    def unexpected_eda_lookup(_name: str) -> str | None:
        pytest.fail("synth reached EDA discovery after failed report preflight")

    monkeypatch.setattr("torch2rtl.synth.yosys.find_eda_tool", unexpected_eda_lookup)

    result = run_yosys(tmp_path)
    rc = cli.cmd_synth(argparse.Namespace(build_dir=tmp_path))
    output = capsys.readouterr().out

    assert not result.ok
    assert "invalid compile report" in result.message
    assert rc == 1
    assert "invalid compile report" in output
    assert "Traceback" not in output
    recorded = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert recorded["preflight_failure"]["section"] == "synthesis"
    assert recorded["status"]["synthesis"]["ok"] is False


@pytest.mark.parametrize("section", ["simulation", "synthesis"])
@pytest.mark.parametrize(
    "malformed_field",
    ["status", "generated-files-number", "generated-files-item-number"],
)
def test_malformed_report_replaces_stale_pass_in_report_and_visualization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    malformed_field: str,
) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    if section == "simulation":
        passed: dict[str, object] = SimulationResult(
            True,
            "passed",
            "simulation passed",
            stdout="PASS vectors=2\n",
        ).__dict__
    else:
        passed = SynthResult(
            True,
            "passed",
            "synthesis passed",
            stdout="Number of cells: 1\n",
        ).__dict__
    update_report(tmp_path, section, passed)

    report_path = tmp_path / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if malformed_field == "status":
        report["status"] = 1
    elif malformed_field == "generated-files-number":
        report["generated_files"] = 1
    else:
        report["generated_files"] = ["top.sv", 1]
    report_path.write_text(json.dumps(report), encoding="utf-8")

    if section == "simulation":
        monkeypatch.setattr(
            "torch2rtl.verify.simulator.find_eda_tool",
            lambda _name: pytest.fail("verify reached EDA after failed preflight"),
        )
        result = run_simulation(tmp_path)
    else:
        monkeypatch.setattr(
            "torch2rtl.synth.yosys.find_eda_tool",
            lambda _name: pytest.fail("synth reached EDA after failed preflight"),
        )
        result = run_yosys(tmp_path)

    assert not result.ok
    repaired_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert repaired_report["preflight_failure"]["section"] == section
    assert repaired_report["status"][section]["ok"] is False
    assert repaired_report[section]["ok"] is False

    manifest = json.loads(
        (tmp_path / "visualization.json").read_text(encoding="utf-8")
    )
    assert manifest["build"]["status"][section]["ok"] is False
    assert manifest["build"][section]["ok"] is False
    html = (tmp_path / "visualization.html").read_text(encoding="utf-8")
    assert f'"message": "{section} passed"' not in html
    if section == "simulation":
        assert '"stdout": "PASS vectors=2\\n"' not in html


@pytest.mark.parametrize("section", ["simulation", "synthesis"])
@pytest.mark.parametrize("payload_text", _EXTREME_JSON_PAYLOADS)
def test_extreme_compile_report_json_is_controlled_before_eda_for_api_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    section: str,
    payload_text: str,
) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    if section == "simulation":
        passed: dict[str, object] = SimulationResult(
            True,
            "passed",
            "simulation passed",
            stdout="PASS vectors=2\n",
        ).__dict__
        monkeypatch.setattr(
            "torch2rtl.verify.simulator.find_eda_tool",
            lambda _name: pytest.fail("verify reached EDA after failed preflight"),
        )
        run_api = run_simulation
        run_cli = lambda: cli.cmd_verify(argparse.Namespace(build_dir=tmp_path))
    else:
        passed = SynthResult(
            True,
            "passed",
            "synthesis passed",
            stdout="Number of cells: 1\n",
        ).__dict__
        monkeypatch.setattr(
            "torch2rtl.synth.yosys.find_eda_tool",
            lambda _name: pytest.fail("synth reached EDA after failed preflight"),
        )
        run_api = run_yosys
        run_cli = lambda: cli.cmd_synth(argparse.Namespace(build_dir=tmp_path))
    update_report(tmp_path, section, passed)

    report_path = tmp_path / "report.json"
    report_path.write_text(payload_text, encoding="utf-8")
    result = run_api(tmp_path)

    assert not result.ok
    assert "invalid compile report" in result.message
    recorded = json.loads(report_path.read_text(encoding="utf-8"))
    assert recorded["preflight_failure"]["section"] == section
    assert recorded["status"][section]["ok"] is False
    assert recorded[section]["ok"] is False

    manifest = json.loads(
        (tmp_path / "visualization.json").read_text(encoding="utf-8")
    )
    assert manifest["build"]["status"][section]["ok"] is False
    assert manifest["build"][section]["ok"] is False
    html = (tmp_path / "visualization.html").read_text(encoding="utf-8")
    assert f'"message": "{section} passed"' not in html
    assert '"stdout": "PASS vectors=2\\n"' not in html

    report_path.write_text(payload_text, encoding="utf-8")
    rc = run_cli()
    output = capsys.readouterr().out

    assert rc == 1
    assert "invalid compile report" in output
    assert "Traceback" not in output
    recorded = json.loads(report_path.read_text(encoding="utf-8"))
    assert recorded["preflight_failure"]["section"] == section
    assert recorded["status"][section]["ok"] is False


@pytest.mark.parametrize("section", ["simulation", "synthesis"])
@pytest.mark.parametrize("payload_text", _EXTREME_JSON_PAYLOADS)
def test_extreme_vectors_json_is_controlled_before_eda_for_api_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    section: str,
    payload_text: str,
) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    if section == "simulation":
        passed: dict[str, object] = SimulationResult(
            True,
            "passed",
            "simulation passed",
            stdout="PASS vectors=2\n",
        ).__dict__
        expected_message = "invalid vector metadata"
        monkeypatch.setattr(
            "torch2rtl.verify.simulator.find_eda_tool",
            lambda _name: pytest.fail("verify reached EDA after failed preflight"),
        )
        run_api = run_simulation
        run_cli = lambda: cli.cmd_verify(argparse.Namespace(build_dir=tmp_path))
    else:
        passed = SynthResult(
            True,
            "passed",
            "synthesis passed",
            stdout="Number of cells: 1\n",
        ).__dict__
        expected_message = "invalid synthesis compile contract"
        monkeypatch.setattr(
            "torch2rtl.synth.yosys.find_eda_tool",
            lambda _name: pytest.fail("synth reached EDA after failed preflight"),
        )
        run_api = run_yosys
        run_cli = lambda: cli.cmd_synth(argparse.Namespace(build_dir=tmp_path))
    update_report(tmp_path, section, passed)

    vectors_path = tmp_path / "vectors.json"
    vectors_path.write_text(payload_text, encoding="utf-8")
    result = run_api(tmp_path)

    assert not result.ok
    assert expected_message in result.message
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["status"][section]["ok"] is False
    assert report[section]["ok"] is False
    manifest = json.loads(
        (tmp_path / "visualization.json").read_text(encoding="utf-8")
    )
    assert manifest["build"]["status"][section]["ok"] is False
    assert manifest["build"][section]["ok"] is False
    html = (tmp_path / "visualization.html").read_text(encoding="utf-8")
    assert f'"message": "{section} passed"' not in html
    assert '"stdout": "PASS vectors=2\\n"' not in html

    rc = run_cli()
    output = capsys.readouterr().out

    assert rc == 1
    assert expected_message in output
    assert "Traceback" not in output
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["status"][section]["ok"] is False


@pytest.mark.parametrize(
    ("filename", "declaration", "replacement", "verify_message"),
    [
        (
            "tb_top.sv",
            "localparam int DATA_BITS = 8;",
            f"localparam int DATA_BITS = {_HUGE_JSON_INTEGER};",
            "invalid testbench contract",
        ),
        (
            "top.sv",
            "parameter int DATA_BITS = 8,",
            f"parameter int DATA_BITS = {_HUGE_JSON_INTEGER},",
            "invalid compiled top contract",
        ),
    ],
    ids=["testbench-localparam", "top-parameter"],
)
def test_extreme_systemverilog_contract_integer_is_controlled_before_eda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    declaration: str,
    replacement: str,
    verify_message: str,
) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    source_path = tmp_path / filename
    source_path.write_text(
        source_path.read_text(encoding="utf-8").replace(declaration, replacement, 1),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["rtl_sha256"][filename] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(
        "torch2rtl.verify.simulator.find_eda_tool",
        lambda _name: pytest.fail("verify reached EDA after failed RTL preflight"),
    )
    monkeypatch.setattr(
        "torch2rtl.synth.yosys.find_eda_tool",
        lambda _name: pytest.fail("synth reached EDA after failed RTL preflight"),
    )

    verification = run_simulation(tmp_path)
    synthesis = run_yosys(tmp_path)

    assert not verification.ok
    assert verification.status == "invalid_vectors"
    assert verify_message in verification.message
    assert not synthesis.ok
    assert synthesis.status == "invalid_sources"
    assert "invalid synthesis compile contract" in synthesis.message


def test_vector_whitespace_layout_does_not_break_post_simulation_refresh(
    tmp_path: Path,
) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    path = tmp_path / "input_vectors.txt"
    path.write_text("\n".join(path.read_text().split()) + "\n", encoding="utf-8")

    result = run_simulation(tmp_path)

    assert result.ok
    manifest = json.loads(
        (tmp_path / "visualization.json").read_text(encoding="utf-8")
    )
    assert manifest["build"]["status"]["simulation"]["status"] == "passed"


def test_invalid_vectors_replace_stale_visualization_pass_status(tmp_path: Path) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    assert run_simulation(tmp_path).ok
    (tmp_path / "input_vectors.txt").write_text("", encoding="utf-8")

    result = run_simulation(tmp_path)

    assert not result.ok
    assert result.status == "invalid_vectors"
    manifest = json.loads(
        (tmp_path / "visualization.json").read_text(encoding="utf-8")
    )
    assert manifest["build"]["status"]["simulation"]["status"] == "invalid_vectors"
    assert manifest["trace"]["matched"] is False
    html = (tmp_path / "visualization.html").read_text(encoding="utf-8")
    assert '"status": "invalid_vectors"' in html
    assert '"message": "simulation passed"' not in html


def test_simulation_rejects_non_object_vector_metadata_and_records_failure(
    tmp_path: Path,
) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    (tmp_path / "vectors.json").write_text("[]", encoding="utf-8")

    result = run_simulation(tmp_path)

    assert not result.ok
    assert result.status == "invalid_vectors"
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["simulation"]["status"] == "invalid_vectors"


@pytest.mark.parametrize(
    "filename",
    [
        "vectors.json",
        "input_vectors.txt",
        "expected_classes.txt",
        "expected_logits.txt",
    ],
)
def test_simulation_rejects_invalid_utf8_vector_artifacts(
    tmp_path: Path,
    filename: str,
) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    (tmp_path / filename).write_bytes(b"\xff\xfe")

    result = run_simulation(tmp_path)

    assert not result.ok
    assert result.status == "invalid_vectors"


def test_simulation_pass_marker_requires_nonzero_exact_vector_count() -> None:
    assert not _simulation_ok(0, "PASS vectors=0\n")
    assert not _simulation_ok(0, "PASS vectors=1\n", expected_vectors=2)
    assert _simulation_ok(0, "PASS vectors=2\n", expected_vectors=2)


def test_generated_conv_rtl_matches_python_on_directed_boundaries(
    tmp_path: Path,
) -> None:
    model = nn.Sequential(
        nn.Conv2d(1, 3, kernel_size=2, stride=2, padding=1),
    ).eval()
    with torch.no_grad():
        model[0].weight.copy_(
            torch.tensor(
                [
                    [[[127 / 64, 127 / 64], [127 / 64, 127 / 64]]],
                    [[[-2.0, -2.0], [-2.0, -2.0]]],
                    [[[1.0, -1.0], [1.0, -1.0]]],
                ]
            )
        )
        model[0].bias.copy_(torch.tensor([127 / 64, -2.0, 0.0]))
    graph = parse_model(model, input_shape=(1, 3, 3))
    cfg = FixedPointConfig(bits=8, frac_bits=6, acc_bits=24)
    inputs = np.asarray(
        [
            [127] * 9,
            [-128] * 9,
            [127, -128, 0, -1, 1, 64, -64, 126, -127],
            [0] * 9,
            [-128, 127, -128, 127, -128, 127, -128, 127, -128],
        ],
        dtype=np.int64,
    )
    emit_systemverilog(graph, cfg, tmp_path, vector_count=len(inputs), seed=0)
    qgraph = quantize_graph(graph, cfg)
    results = [infer_quantized(qgraph, row) for row in inputs]
    logits = np.stack([result.logits.reshape(-1) for result in results])
    classes = np.asarray([result.class_id for result in results], dtype=np.int64)
    np.savetxt(tmp_path / "input_vectors.txt", inputs, fmt="%d")
    np.savetxt(tmp_path / "expected_logits.txt", logits, fmt="%d")
    np.savetxt(tmp_path / "expected_classes.txt", classes.reshape(-1, 1), fmt="%d")

    simulation = run_simulation(tmp_path)

    assert simulation.ok, simulation.stdout + simulation.stderr
    assert "PASS vectors=5" in simulation.stdout
    assert np.any(logits == cfg.min_int)
    assert np.any(logits == cfg.max_int)


def test_signed_32_bit_data_endpoints_match_generated_rtl(tmp_path: Path) -> None:
    model = nn.Sequential(nn.Linear(1, 1, bias=False)).eval()
    with torch.no_grad():
        model[0].weight.fill_(1.0)
    cfg = FixedPointConfig(bits=32, frac_bits=0, acc_bits=64)
    graph = parse_model(model, input_shape=(1,))
    inputs = np.asarray([[cfg.min_int], [cfg.max_int]], dtype=np.int64)
    emit_systemverilog(graph, cfg, tmp_path, vector_count=2, seed=0)
    qgraph = quantize_graph(graph, cfg)
    results = [infer_quantized(qgraph, row) for row in inputs]
    logits = np.stack([result.logits.reshape(-1) for result in results])
    classes = np.asarray([result.class_id for result in results], dtype=np.int64)
    np.savetxt(tmp_path / "input_vectors.txt", inputs, fmt="%d")
    np.savetxt(tmp_path / "expected_logits.txt", logits, fmt="%d")
    np.savetxt(tmp_path / "expected_classes.txt", classes.reshape(-1, 1), fmt="%d")

    simulation = run_simulation(tmp_path)

    assert simulation.ok, simulation.stdout + simulation.stderr
    assert "PASS vectors=2" in simulation.stdout
    np.testing.assert_array_equal(logits.reshape(-1), [cfg.min_int, cfg.max_int])
