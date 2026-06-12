from __future__ import annotations

import torch.nn as nn

from torch2rtl.frontend.pytorch_fx import parse_model
from torch2rtl.ir.ops import ArgmaxIR, LinearIR, ReluIR


def test_fx_parser_extracts_linear_relu_linear() -> None:
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 4),
    )
    graph = parse_model(model, input_shape=(16,))
    assert [type(op) for op in graph.ops] == [LinearIR, ReluIR, LinearIR, ArgmaxIR]
    assert graph.linear_ops[0].in_features == 16
    assert graph.linear_ops[1].out_features == 4
