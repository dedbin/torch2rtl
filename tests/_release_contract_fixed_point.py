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



@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.asarray([np.iinfo(np.uint64).max], dtype=np.uint64), "range"),
        (np.asarray([True], dtype=np.bool_), "boolean"),
        (np.asarray([1.9], dtype=np.float64), "integer"),
        (np.asarray([np.nan], dtype=np.float64), "finite"),
        (np.asarray([np.inf], dtype=np.float64), "finite"),
        (np.asarray([2**100], dtype=object), "object"),
    ],
)
@pytest.mark.parametrize("position", ["input", "weight", "bias"])
def test_linear_raw_fixed_values_are_validated_before_cast(
    value: np.ndarray,
    message: str,
    position: str,
) -> None:
    inputs: object = np.asarray([1], dtype=np.int64)
    weight: object = np.asarray([[1]], dtype=np.int64)
    bias: object = np.asarray([0], dtype=np.int64)
    if position == "input":
        inputs = value
    elif position == "weight":
        weight = value.reshape(1, 1)
    else:
        bias = value

    with pytest.raises((TypeError, ValueError, OverflowError), match=message):
        linear_fixed(inputs, weight, bias, FixedPointConfig())


@pytest.mark.parametrize("position", ["input", "weight", "bias"])
def test_conv_raw_fixed_values_are_validated_before_cast(position: str) -> None:
    op = QuantizedConv2dIR(
        name="conv",
        in_channels=1,
        out_channels=1,
        input_height=1,
        input_width=1,
        output_height=1,
        output_width=1,
        kernel_height=1,
        kernel_width=1,
        stride=(1, 1),
        padding=(0, 0),
        weight=np.asarray([[[[1]]]], dtype=np.int64),
        bias=np.asarray([0], dtype=np.int64),
    )
    inputs: object = np.asarray([1], dtype=np.int64)
    if position == "input":
        inputs = np.asarray([1.5])
    elif position == "weight":
        object.__setattr__(op, "weight", np.asarray([[[[True]]]]))
    else:
        object.__setattr__(op, "bias", np.asarray([np.inf]))

    with pytest.raises((TypeError, ValueError), match="integer|boolean|finite"):
        conv2d_fixed(inputs, op, FixedPointConfig())


def test_accumulator_analysis_rejects_invalid_raw_parameters_before_cast() -> None:
    qgraph = QuantizedGraph(
        input_shape=(1,),
        cfg=FixedPointConfig(),
        ops=(
            QuantizedLinearIR(
                name="invalid",
                in_features=1,
                out_features=1,
                weight=np.asarray([[np.iinfo(np.uint64).max]], dtype=np.uint64),
                bias=np.asarray([0], dtype=np.int64),
            ),
        ),
    )

    with pytest.raises(OverflowError, match="range"):
        validate_accumulator_width(qgraph)


@pytest.mark.parametrize("bits", [2, 32])
def test_fixed_point_bits_supported_boundaries(bits: int) -> None:
    cfg = FixedPointConfig(bits=bits, frac_bits=bits - 1, acc_bits=64)
    assert cfg.bits == bits


@pytest.mark.parametrize("bits", [1, 33])
def test_fixed_point_bits_rejects_outside_supported_boundaries(bits: int) -> None:
    with pytest.raises(ValueError, match="bits"):
        FixedPointConfig(bits=bits, frac_bits=0, acc_bits=64)


@pytest.mark.parametrize("frac_bits", [0, 7])
def test_fixed_point_frac_bits_supported_boundaries(frac_bits: int) -> None:
    assert FixedPointConfig(bits=8, frac_bits=frac_bits, acc_bits=9).frac_bits == frac_bits


@pytest.mark.parametrize("frac_bits", [-1, 8])
def test_fixed_point_frac_bits_rejects_outside_supported_boundaries(
    frac_bits: int,
) -> None:
    with pytest.raises(ValueError, match="frac_bits"):
        FixedPointConfig(bits=8, frac_bits=frac_bits, acc_bits=9)


@pytest.mark.parametrize("acc_bits", [9, 64])
def test_fixed_point_acc_bits_supported_boundaries(acc_bits: int) -> None:
    assert FixedPointConfig(bits=8, frac_bits=0, acc_bits=acc_bits).acc_bits == acc_bits


@pytest.mark.parametrize("acc_bits", [8, 65])
def test_fixed_point_acc_bits_rejects_outside_supported_boundaries(acc_bits: int) -> None:
    with pytest.raises(ValueError, match="acc_bits"):
        FixedPointConfig(bits=8, frac_bits=0, acc_bits=acc_bits)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bits": True},
        {"bits": 8.0},
        {"frac_bits": True},
        {"frac_bits": 6.0},
        {"acc_bits": True},
        {"acc_bits": 32.0},
    ],
)
def test_fixed_point_config_rejects_non_integer_field_types(
    kwargs: dict[str, object],
) -> None:
    values: dict[str, object] = {"bits": 8, "frac_bits": 6, "acc_bits": 32}
    values.update(kwargs)
    with pytest.raises(TypeError):
        FixedPointConfig(**values)  # type: ignore[arg-type]


def test_quantize_large_finite_float32_saturates_without_warning() -> None:
    cfg = FixedPointConfig(bits=32, frac_bits=31, acc_bits=64)
    values = np.asarray(
        [-np.finfo(np.float32).max, np.finfo(np.float32).max],
        dtype=np.float32,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = quantize_array(values, cfg)

    np.testing.assert_array_equal(result, [cfg.min_int, cfg.max_int])
