from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch.nn as nn

from torch2rtl.backend.systemverilog.emit import emit_systemverilog
from torch2rtl.frontend.pytorch_fx import parse_model
from torch2rtl.quant.fixed_point import FixedPointConfig
from torch2rtl.synth.report import update_report


def _emit_tiny_mlp(tmp_path: Path, vector_count: int = 3) -> Path:
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 4),
    )
    graph = parse_model(model, input_shape=(16,))
    cfg = FixedPointConfig(bits=8, frac_bits=6, acc_bits=32)
    emit_systemverilog(graph, cfg, tmp_path, vector_count=vector_count, seed=1)
    return tmp_path


def test_visualization_manifest_contains_circuit_blocks(tmp_path: Path) -> None:
    build_dir = _emit_tiny_mlp(tmp_path)
    manifest = json.loads((build_dir / "visualization.json").read_text(encoding="utf-8"))

    main_kinds = [
        block["kind"]
        for block in manifest["blocks"]
        if block["lane"] == "main"
    ]
    assert main_kinds == ["input", "linear", "relu", "linear", "argmax", "output"]
    assert manifest["quant"] == {"bits": 8, "frac_bits": 6, "acc_bits": 32}
    assert manifest["trace"]["vector_index"] == 0
    assert manifest["trace"]["logits"]["size"] == 4

    first_linear = next(block for block in manifest["blocks"] if block["kind"] == "linear")
    assert "Torch2RTL" in manifest["title"]
    assert manifest["blocks"][0]["name"] == "in_data"
    assert first_linear["signal_in"] == "in_data"
    assert first_linear["in_features"] == 16
    assert first_linear["out_features"] == 32


def test_visualization_html_is_self_contained(tmp_path: Path) -> None:
    build_dir = _emit_tiny_mlp(tmp_path)
    html = (build_dir / "visualization.html").read_text(encoding="utf-8")

    assert 'id="manifest-data"' in html
    assert 'id="circuit-root"' in html
    assert "Linear / MAC" in html
    assert "Torch2RTL" in html
    assert "visualization.json" in html


def test_visualize_cli_renders_from_existing_build(tmp_path: Path) -> None:
    build_dir = _emit_tiny_mlp(tmp_path, vector_count=4)
    out_path = tmp_path / "custom_visualization.html"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch2rtl.cli",
            "visualize",
            str(build_dir),
            "--out",
            str(out_path),
            "--vector-index",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert out_path.exists()
    html = out_path.read_text(encoding="utf-8")
    assert '"vector_index": 2' in html
    assert "Torch2RTL" in html


def test_update_report_refreshes_visualization_status(tmp_path: Path) -> None:
    build_dir = _emit_tiny_mlp(tmp_path)
    manifest_before = json.loads(
        (build_dir / "visualization.json").read_text(encoding="utf-8")
    )
    assert manifest_before["build"]["status"] == {}

    update_report(
        build_dir,
        "simulation",
        {
            "ok": True,
            "status": "passed",
            "message": "simulation passed",
            "stdout": "PASS vectors=3\n",
            "stderr": "",
        },
    )

    manifest_after = json.loads(
        (build_dir / "visualization.json").read_text(encoding="utf-8")
    )
    html = (build_dir / "visualization.html").read_text(encoding="utf-8")

    assert manifest_after["build"]["status"]["simulation"]["status"] == "passed"
    assert "simulation passed" in html
