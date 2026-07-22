from __future__ import annotations

from pathlib import Path

import numpy as np
import torch.nn as nn

from torch2rtl.backend.systemverilog.emit import emit_systemverilog
from torch2rtl.frontend.pytorch_fx import parse_model
from torch2rtl.quant.fixed_point import FixedPointConfig


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
    assert 'while ($fscanf(fd_expected, "%d", expected_class) == 1)' in tb_top
    assert "while (!$feof" not in tb_top


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
