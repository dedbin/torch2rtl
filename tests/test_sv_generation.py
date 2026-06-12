from __future__ import annotations

from pathlib import Path

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
    assert 'while ($fscanf(fd_expected, "%d", expected_class) == 1)' in tb_top
    assert "while (!$feof" not in tb_top
