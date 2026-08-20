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



def test_float_reference_rejects_string_input_instead_of_casting() -> None:
    graph = parse_model(nn.Sequential(nn.ReLU()), input_shape=(2,))

    with pytest.raises(TypeError, match="float32/float64 or integer"):
        infer_float_graph(graph, np.asarray(["1.25", "-0.5"]))


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("kernel_size", (0, 1)),
        ("kernel_size", (-1, 1)),
        ("kernel_size", (True, 1)),
        ("kernel_size", (1.5, 1)),
        ("stride", (0, 1)),
        ("stride", (-1, 1)),
        ("stride", (True, 1)),
        ("padding", (-1, 0)),
        ("padding", (True, 0)),
    ],
)
def test_frontend_rejects_invalid_conv_parameters(
    attribute: str,
    value: object,
) -> None:
    conv = nn.Conv2d(1, 1, kernel_size=1)
    setattr(conv, attribute, value)

    with pytest.raises(UnsupportedOpError, match=attribute):
        parse_model(nn.Sequential(conv), input_shape=(1, 4, 4))


@pytest.mark.parametrize(
    ("attribute", "value"),
    [("start_dim", None), ("start_dim", True), ("end_dim", "-1")],
)
def test_frontend_rejects_non_exact_flatten_dimensions(
    attribute: str,
    value: object,
) -> None:
    flatten = nn.Flatten()
    setattr(flatten, attribute, value)

    with pytest.raises(UnsupportedOpError, match=attribute):
        parse_model(nn.Sequential(flatten), input_shape=(2, 2))


def test_frontend_rejects_input_shape_integer_subclasses() -> None:
    class ShapeDimension(int):
        pass

    with pytest.raises(UnsupportedOpError, match="input shape"):
        parse_model(nn.Sequential(nn.ReLU()), input_shape=(ShapeDimension(2),))


def test_frontend_rejects_hostile_input_shape_without_raw_exception() -> None:
    class HostileDimension:
        def __repr__(self) -> str:
            raise ValueError("shape repr trap")

    class HostileShape:
        def __iter__(self) -> object:
            raise ValueError("shape iteration trap")

    model = nn.Sequential(nn.ReLU())
    with pytest.raises(UnsupportedOpError, match="input shape"):
        parse_model(model, input_shape=(HostileDimension(),))  # type: ignore[arg-type]
    with pytest.raises(UnsupportedOpError, match="input shape"):
        parse_model(model, input_shape=HostileShape())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "input_shape",
    [
        {2},
        iter([2]),
        b"2",
    ],
)
def test_frontend_rejects_non_sequence_shape_containers(input_shape: object) -> None:
    with pytest.raises(UnsupportedOpError, match="input shape"):
        parse_model(  # type: ignore[arg-type]
            nn.Sequential(nn.ReLU()),
            input_shape=input_shape,
        )


@pytest.mark.parametrize("dimension", [2**63, 10**100])
def test_frontend_rejects_unrepresentable_input_dimensions(dimension: int) -> None:
    with pytest.raises(UnsupportedOpError, match="input shape"):
        parse_model(nn.Sequential(nn.ReLU()), input_shape=(dimension,))


def test_frontend_rejects_input_rank_above_numpy_limit() -> None:
    with pytest.raises(UnsupportedOpError, match="input shape"):
        parse_model(nn.Sequential(nn.ReLU()), input_shape=(1,) * 65)


def test_frontend_rejects_conv_structural_int_outside_signed_32_bit() -> None:
    stride = 5_000_000_000
    padding = 2 * stride - 2**32
    conv = nn.Conv2d(1, 1, kernel_size=1)
    conv.stride = (stride, stride)
    conv.padding = (padding, padding)

    with pytest.raises(UnsupportedOpError, match="signed 32-bit"):
        parse_model(nn.Sequential(conv), input_shape=(1, 1, 1))


def test_frontend_rejects_oversized_conv_output_before_probe() -> None:
    conv = nn.Conv2d(1, 1, kernel_size=1, padding=50_000)

    with pytest.raises(UnsupportedOpError, match="element count"):
        parse_model(nn.Sequential(conv), input_shape=(1, 1, 1))


def test_frontend_rejects_conv_attributes_inconsistent_with_parameter_shapes() -> None:
    conv = nn.Conv2d(1, 1, kernel_size=1)
    conv.kernel_size = (2, 2)

    with pytest.raises(UnsupportedOpError, match="weight shape"):
        parse_model(nn.Sequential(conv), input_shape=(1, 4, 4))


@pytest.mark.parametrize(
    ("module", "input_shape"),
    [
        (nn.Linear(2, 2), (2,)),
        (nn.Conv2d(1, 1, kernel_size=1), (1, 2, 2)),
    ],
)
def test_frontend_rejects_missing_required_weight_parameter(
    module: nn.Module,
    input_shape: tuple[int, ...],
) -> None:
    module.register_parameter("weight", None)

    with pytest.raises(UnsupportedOpError, match="weight"):
        parse_model(nn.Sequential(module), input_shape=input_shape)


@pytest.mark.parametrize(
    ("module", "attribute", "input_shape"),
    [
        (nn.Linear(2, 2), "weight", (2,)),
        (nn.Conv2d(1, 1, kernel_size=1), "stride", (1, 2, 2)),
        (nn.ReLU(), "inplace", (2,)),
        (nn.Flatten(), "start_dim", (2, 2)),
    ],
)
def test_frontend_wraps_missing_standard_module_attributes(
    module: nn.Module,
    attribute: str,
    input_shape: tuple[int, ...],
) -> None:
    delattr(module, attribute)

    with pytest.raises(UnsupportedOpError, match=attribute):
        parse_model(nn.Sequential(module), input_shape=input_shape)


@pytest.mark.parametrize(
    ("module", "input_shape", "message"),
    [
        (nn.Linear(2, 0), (2,), "out_features"),
        (nn.Conv2d(1, 0, kernel_size=1), (1, 2, 2), "out_channels"),
    ],
)
def test_frontend_rejects_zero_sized_layer_outputs(
    module: nn.Module,
    input_shape: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(UnsupportedOpError, match=message):
        parse_model(nn.Sequential(module), input_shape=input_shape)


@pytest.mark.parametrize(
    ("module", "attribute", "value", "input_shape"),
    [
        (nn.Linear(2, 2), "in_features", True, (2,)),
        (nn.Linear(2, 2), "out_features", "2", (2,)),
        (nn.Conv2d(1, 1, kernel_size=1), "in_channels", True, (1, 2, 2)),
        (nn.Conv2d(1, 1, kernel_size=1), "out_channels", "1", (1, 2, 2)),
        (nn.Conv2d(1, 1, kernel_size=1), "groups", True, (1, 2, 2)),
    ],
)
def test_frontend_rejects_non_exact_structural_integer_attributes(
    module: nn.Module,
    attribute: str,
    value: object,
    input_shape: tuple[int, ...],
) -> None:
    setattr(module, attribute, value)

    with pytest.raises(UnsupportedOpError, match=attribute):
        parse_model(nn.Sequential(module), input_shape=input_shape)
