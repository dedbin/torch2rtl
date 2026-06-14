from __future__ import annotations

import pytest
import torch.nn as nn

from torch2rtl.frontend.pytorch_fx import UnsupportedOpError, parse_model
from torch2rtl.ir.ops import ArgmaxIR, Conv2dIR, FlattenIR, LinearIR, ReluIR


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


def test_fx_parser_extracts_conv2d_relu_flatten_linear() -> None:
    model = nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(start_dim=0),
        nn.Linear(8, 3),
    )
    graph = parse_model(model, input_shape=(1, 4, 4))

    assert [type(op) for op in graph.ops] == [
        Conv2dIR,
        ReluIR,
        FlattenIR,
        LinearIR,
        ArgmaxIR,
    ]
    conv = graph.ops[0]
    assert isinstance(conv, Conv2dIR)
    assert conv.in_channels == 1
    assert conv.out_channels == 2
    assert conv.output.shape == (2, 2, 2)
    assert conv.stride == (2, 2)
    assert conv.padding == (1, 1)
    assert graph.linear_ops[0].in_features == 8


def test_fx_parser_rejects_conv2d_batch_dimension() -> None:
    model = nn.Sequential(nn.Conv2d(1, 2, kernel_size=3))

    with pytest.raises(UnsupportedOpError, match="batch dimension"):
        parse_model(model, input_shape=(1, 1, 4, 4))


def test_fx_parser_rejects_grouped_conv2d() -> None:
    model = nn.Sequential(nn.Conv2d(2, 2, kernel_size=3, groups=2))

    with pytest.raises(UnsupportedOpError, match="groups=1"):
        parse_model(model, input_shape=(2, 4, 4))


def test_fx_parser_rejects_dilated_conv2d() -> None:
    model = nn.Sequential(nn.Conv2d(1, 2, kernel_size=3, dilation=2))

    with pytest.raises(UnsupportedOpError, match="dilation=1"):
        parse_model(model, input_shape=(1, 6, 6))
