from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from torch2rtl.backend.systemverilog.emit import emit_systemverilog
from torch2rtl.frontend.pytorch_fx import parse_model
from torch2rtl.quant.fixed_point import FixedPointConfig
from torch2rtl.quant.reference import infer_quantized, quantize_graph
from torch2rtl.verify.simulator import run_simulation


def test_generated_sv_contains_expected_module_names(tmp_path: Path) -> None:
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 4),
    )
    graph = parse_model(model, input_shape=(16,))
    cfg = FixedPointConfig(bits=8, frac_bits=6, acc_bits=32)
    report = emit_systemverilog(graph, cfg, tmp_path, vector_count=2, seed=1)

    assert "expected_logits.txt" in report.generated_files

    expected_logits = np.loadtxt(tmp_path / "expected_logits.txt", dtype=np.int64)
    assert expected_logits.shape == (2, 4)

    assert "top.sv" in report.generated_files
    assert "tb_top.sv" in report.generated_files
    assert (tmp_path / "top.sv").exists()
    assert (tmp_path / "report.json").exists()
    combined = "\n".join(
        (tmp_path / name).read_text(encoding="utf-8")
        for name in ["linear_comb.sv", "relu.sv", "argmax.sv", "top.sv"]
    )
    assert "module linear_comb" in combined
    assert "module relu" in combined
    assert "module argmax" in combined
    assert "module top" in combined
    tb_top = (tmp_path / "tb_top.sv").read_text(encoding="utf-8")
    assert '$fopen("expected_logits.txt", "r")' in tb_top
    assert "actual_logit !== expected_logit" in tb_top
    assert "localparam int EXPECTED_VECTORS = 2" in tb_top
    assert "vector_idx < EXPECTED_VECTORS" in tb_top
    assert "extra or malformed expected logit data" in tb_top


def test_generated_sv_contains_conv2d_backend(tmp_path: Path) -> None:
    model = nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(start_dim=0),
        nn.Linear(32, 4),
    )
    graph = parse_model(model, input_shape=(1, 4, 4))
    cfg = FixedPointConfig(bits=8, frac_bits=6, acc_bits=32)
    report = emit_systemverilog(graph, cfg, tmp_path, vector_count=2, seed=3)

    assert "conv2d_comb.sv" in report.generated_files
    assert (tmp_path / "conv2d_comb.sv").exists()
    top = (tmp_path / "top.sv").read_text(encoding="utf-8")
    payload = (tmp_path / "report.json").read_text(encoding="utf-8")

    assert "module conv2d_comb" in (tmp_path / "conv2d_comb.sv").read_text(
        encoding="utf-8"
    )
    assert ".IN_CHANNELS(1)" in top
    assert ".OUT_CHANNELS(2)" in top
    assert ".OUT_HEIGHT(4)" in top
    assert '"Conv2dIR"' in payload
    assert '"macs"' in payload
    assert '"fixed_vs_float_ir"' in payload


def test_emit_systemverilog_removes_stale_generated_artifacts(tmp_path: Path) -> None:
    (tmp_path / "old_weights.mem").write_text("1\n", encoding="utf-8")
    (tmp_path / "old_bias.mem").write_text("1\n", encoding="utf-8")
    (tmp_path / "simv").write_text("stale\n", encoding="utf-8")

    model = nn.Sequential(
        nn.Linear(4, 2),
        nn.ReLU(),
    )
    graph = parse_model(model, input_shape=(4,))
    cfg = FixedPointConfig(bits=8, frac_bits=6, acc_bits=32)
    emit_systemverilog(graph, cfg, tmp_path, vector_count=2, seed=4)

    assert not (tmp_path / "old_weights.mem").exists()
    assert not (tmp_path / "old_bias.mem").exists()
    assert not (tmp_path / "simv").exists()


def test_emit_rejects_unsafe_accumulator_before_writing_rtl(tmp_path: Path) -> None:
    model = nn.Sequential(nn.Linear(1, 1, bias=False))
    with torch.no_grad():
        model[0].weight.fill_(127 / 64)
    graph = parse_model(model, input_shape=(1,))
    cfg = FixedPointConfig(bits=8, frac_bits=6, acc_bits=14)

    with pytest.raises(OverflowError, match="requires at least 15"):
        emit_systemverilog(graph, cfg, tmp_path, vector_count=1, seed=0)

    assert not (tmp_path / "top.sv").exists()


def test_generated_rtl_matches_python_on_directed_boundary_vectors(
    tmp_path: Path,
) -> None:
    model = nn.Sequential(nn.Linear(2, 3, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(
            torch.tensor(
                [
                    [127 / 64, 127 / 64],
                    [127 / 64, -127 / 64],
                    [-127 / 64, 127 / 64],
                ]
            )
        )
    graph = parse_model(model, input_shape=(2,))
    cfg = FixedPointConfig(bits=8, frac_bits=6, acc_bits=16)
    inputs = np.asarray(
        [
            [127, 127],
            [-128, -128],
            [127, -128],
            [-128, 127],
            [0, 0],
            [1, -1],
        ],
        dtype=np.int64,
    )
    emit_systemverilog(
        graph,
        cfg,
        tmp_path,
        input_vectors=inputs,
        vector_source="directed_boundary_test",
    )
    qgraph = quantize_graph(graph, cfg)
    results = [infer_quantized(qgraph, row) for row in inputs]
    expected_logits = np.asarray([result.logits for result in results], dtype=np.int64)
    expected_classes = np.asarray(
        [result.class_id for result in results],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(
        np.loadtxt(tmp_path / "input_vectors.txt", dtype=np.int64),
        inputs,
    )
    np.testing.assert_array_equal(
        np.loadtxt(tmp_path / "expected_logits.txt", dtype=np.int64),
        expected_logits,
    )
    np.testing.assert_array_equal(
        np.loadtxt(tmp_path / "expected_classes.txt", dtype=np.int64),
        expected_classes,
    )

    simulation = run_simulation(tmp_path)

    assert simulation.ok, simulation.stdout + simulation.stderr
    assert "PASS vectors=6" in simulation.stdout
    assert np.any(expected_logits == cfg.min_int)
    assert np.any(expected_logits == cfg.max_int)
