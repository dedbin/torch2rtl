from __future__ import annotations

import json
from pathlib import Path

import pytest

from torch2rtl.backend.systemverilog.emit import BuildReport
from torch2rtl.cli import _format_demo_summary, build_parser
from torch2rtl.demo import DemoResult, get_demo_spec, run_demo
from torch2rtl.synth.yosys import SynthResult, parse_yosys_metrics
from torch2rtl.verify.simulator import SimulationResult


def test_demo_parser_accepts_name_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["demo", "--name", "tiny-conv", "--out", "build/demo"])

    assert args.command == "demo"
    assert args.name == "tiny-conv"
    assert args.demo_name is None
    assert args.out == Path("build/demo")


def test_demo_parser_accepts_positional_name() -> None:
    parser = build_parser()
    args = parser.parse_args(["demo", "tiny-conv"])

    assert args.command == "demo"
    assert args.demo_name == "tiny-conv"
    assert args.name is None


def test_get_demo_spec_rejects_unknown_demo() -> None:
    with pytest.raises(ValueError, match="unknown demo"):
        get_demo_spec("missing")


def test_format_demo_summary_lists_demo_artifacts(tmp_path: Path) -> None:
    result = DemoResult(
        name="tiny-conv",
        build_report=BuildReport(
            out_dir=tmp_path,
            generated_files=(
                "argmax.sv",
                "expected_classes.txt",
                "expected_logits.txt",
                "input_vectors.txt",
                "report.json",
                "tb_top.sv",
                "top.sv",
                "vectors.json",
                "visualization.html",
            ),
            report_path=tmp_path / "report.json",
        ),
        simulation=SimulationResult(
            ok=False,
            status="not_found",
            message="simulator not found: install Icarus Verilog or Verilator",
        ),
        synthesis=SynthResult(
            ok=False,
            status="not_found",
            message="yosys not found: install Yosys to run synthesis",
        ),
        visualization_path=tmp_path / "visualization.html",
    )

    lines = _format_demo_summary(
        result,
        tools={"iverilog": False, "vvp": False, "verilator": False, "yosys": False},
    )
    text = "\n".join(lines)

    assert "compile: ok (9 generated artifacts)" in text
    assert "rtl files: argmax.sv, tb_top.sv, top.sv" in text
    assert (
        "test data: expected_classes.txt, expected_logits.txt, input_vectors.txt, "
        "vectors.json" in text
    )
    assert "verification: skipped" in text
    assert "synthesis: skipped" in text
    assert f"html report: {tmp_path / 'visualization.html'}" in text
    assert "optional EDA tools missing: iverilog, vvp, verilator, yosys" in text


def test_parse_yosys_metrics_extracts_design_hierarchy() -> None:
    log = """
=== design hierarchy ===

        +----------Count including submodules.
        |
     4195 wires
  2410258 wire bits
     2736 public wires
  2367860 public wire bits
       15 ports
     6788 port bits
        - memories
        - memory bits
        - processes
     1496 cells
      640   $add
       39   $gt
       68   $lt
      640   $mul
      109   $mux
"""

    metrics = parse_yosys_metrics(log)

    assert metrics["wires"] == 4195
    assert metrics["wire_bits"] == 2410258
    assert metrics["public_wires"] == 2736
    assert metrics["cells"] == 1496
    assert metrics["memories"] == 0
    assert metrics["cell_types"]["$mul"] == 640


def test_run_demo_tiny_conv_creates_board_free_artifacts(tmp_path: Path) -> None:
    result = run_demo("tiny-conv", out_dir=tmp_path, vectors=2, seed=5)

    assert result.name == "tiny-conv"
    assert result.visualization_path == tmp_path / "visualization.html"
    assert (tmp_path / "top.sv").exists()
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "visualization.json").exists()
    assert (tmp_path / "visualization.html").exists()

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "visualization.json").read_text(encoding="utf-8"))
    html = (tmp_path / "visualization.html").read_text(encoding="utf-8")

    assert report["demo"]["name"] == "tiny-conv"
    assert report["demo"]["commands"][0]["step"] == "demo"
    assert "environment" in report
    assert "simulation" in report
    assert "synthesis" in report
    assert manifest["build"]["demo"]["name"] == "tiny-conv"
    assert manifest["build"]["artifacts"]
    assert "Ресурсы синтеза" in html
    assert "Артефакты" in html
