from __future__ import annotations

import builtins
import copy
import dis
import inspect
import math
import pickle
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
import torch.nn as nn

import torch2rtl.frontend.pytorch_fx as fx_frontend
from torch2rtl.frontend.pytorch_fx import UnsupportedOpError, parse_model
from torch2rtl.ir.ops import Conv2dIR, ReluIR
from torch2rtl.quant.reference import infer_float_graph


def _basic_model(
    *,
    in_channels: int = 1,
    out_channels: int = 1,
    conv_bias: bool = True,
    bn_affine: bool = True,
    dtype: torch.dtype = torch.float32,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=conv_bias,
            dtype=dtype,
        ),
        nn.BatchNorm2d(
            out_channels,
            affine=bn_affine,
            track_running_stats=True,
            dtype=dtype,
        ),
    ).eval()


def _tensor_snapshot(tensor: torch.Tensor) -> tuple[object, ...]:
    values = (
        tensor.detach()
        .resolve_conj()
        .resolve_neg()
        .cpu()
        .contiguous()
        .numpy()
    )
    gradient = None
    if tensor.grad is not None:
        gradient = _tensor_snapshot(tensor.grad)
    return (
        id(tensor),
        type(tensor),
        str(tensor.dtype),
        str(tensor.device),
        str(tensor.layout),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.requires_grad,
        tensor.is_conj(),
        tensor.is_neg(),
        gradient,
        values.tobytes(),
    )


def _source_snapshot(model: nn.Module) -> tuple[object, ...]:
    modules = tuple(
        (
            name,
            id(module),
            type(module),
            module.training,
            id(type(module).forward),
            tuple((key, id(value)) for key, value in module._forward_hooks.items()),
            tuple(
                (key, id(value)) for key, value in module._forward_pre_hooks.items()
            ),
            tuple(
                (key, id(value)) for key, value in module._backward_hooks.items()
            ),
        )
        for name, module in model.named_modules(remove_duplicate=False)
    )
    parameters = tuple(
        (name, _tensor_snapshot(parameter))
        for name, parameter in model.named_parameters(remove_duplicate=False)
    )
    buffers = tuple(
        (name, _tensor_snapshot(buffer))
        for name, buffer in model.named_buffers(remove_duplicate=False)
    )
    marker = getattr(model, "marker", None)
    return modules, parameters, buffers, marker



@pytest.mark.parametrize(
    "mode_case",
    ["root", "conv", "batchnorm", "mixed"],
)
def test_rejects_training_state_without_mutating_source(mode_case: str) -> None:
    if mode_case == "mixed":
        model = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
        ).eval()
        model[3].train()
    else:
        model = _basic_model()
        if mode_case == "root":
            model.train()
            model[0].eval()
            model[1].eval()
        elif mode_case == "conv":
            model[0].train()
        else:
            model[1].train()
    before = _source_snapshot(model)

    with pytest.raises(UnsupportedOpError, match="eval|training"):
        parse_model(model, input_shape=(1, 2, 2))

    assert _source_snapshot(model) == before


def test_successful_fusion_preserves_source_identities_values_flags_and_hooks() -> None:
    model = _basic_model(out_channels=2, conv_bias=False)
    model.marker = ("source", 17)
    before = _source_snapshot(model)

    graph = parse_model(model, input_shape=(1, 2, 2))

    assert [type(op) for op in graph.ops] == [Conv2dIR]
    assert _source_snapshot(model) == before


def test_failed_second_pair_preserves_source_after_first_pair_was_eligible() -> None:
    model = nn.Sequential(
        nn.Conv2d(1, 1, 1),
        nn.BatchNorm2d(1),
        nn.Conv2d(1, 1, 1),
        nn.BatchNorm2d(1),
    ).eval()
    assert model[3].running_var is not None
    model[3].running_var.fill_(-1.0)
    before = _source_snapshot(model)

    with pytest.raises(UnsupportedOpError, match="running_var|variance"):
        parse_model(model, input_shape=(1, 2, 2))

    assert _source_snapshot(model) == before


class _StatefulConvBn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.counter = 0
        self.conv = nn.Conv2d(1, 1, 1)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.counter += 1
        return self.bn(self.conv(inputs))


def test_trace_time_python_state_mutation_is_rejected_without_touching_source() -> None:
    model = _StatefulConvBn().eval()
    before = _source_snapshot(model)

    with pytest.raises(UnsupportedOpError, match="state|bytecode"):
        parse_model(model, input_shape=(1, 2, 2))

    assert model.counter == 0
    assert _source_snapshot(model) == before


class _ConvFanout(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        conv_output = self.conv(inputs)
        self.bn(conv_output)
        return self.relu(conv_output)


class _BatchNormFanout(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.first = nn.ReLU()
        self.second = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.bn(self.conv(inputs))
        self.first(normalized)
        return self.second(normalized)


@pytest.mark.parametrize(
    "model",
    [_ConvFanout().eval(), _BatchNormFanout().eval()],
    ids=["conv-fanout", "batchnorm-fanout"],
)
def test_rejects_fanout_around_fusion_pair(model: nn.Module) -> None:
    with pytest.raises(UnsupportedOpError, match="fan-out|user|sequential"):
        parse_model(model, input_shape=(1, 2, 2))


@pytest.mark.parametrize(
    "model",
    [
        nn.Sequential(nn.BatchNorm2d(1)).eval(),
        nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(1),
        ).eval(),
    ],
    ids=["standalone", "not-immediately-after-conv"],
)
def test_rejects_unfusable_batchnorm(model: nn.Module) -> None:
    with pytest.raises(UnsupportedOpError, match="BatchNorm2d|batch.?norm|fusion"):
        parse_model(model, input_shape=(1, 2, 2))


def test_rejects_public_nchw_even_when_pair_is_fusable() -> None:
    with pytest.raises(UnsupportedOpError, match="batch dimension|NCHW|CHW"):
        parse_model(_basic_model(), input_shape=(1, 1, 2, 2))
