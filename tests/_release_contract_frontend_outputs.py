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



@pytest.mark.parametrize("with_relu", [False, True], ids=["conv", "conv-relu"])
def test_terminal_multidimensional_output_is_flattened_only_in_vector_files(
    tmp_path: Path,
    with_relu: bool,
) -> None:
    layers: list[nn.Module] = [nn.Conv2d(1, 1, kernel_size=2)]
    if with_relu:
        layers.append(nn.ReLU())
    model = nn.Sequential(*layers).eval()
    graph = parse_model(model, input_shape=(1, 3, 3))
    inputs = np.arange(9, dtype=np.float32).reshape(1, 3, 3) / 8

    floating = infer_float_graph(graph, inputs)
    qgraph = quantize_graph(graph, FixedPointConfig())
    fixed = infer_quantized(qgraph, quantize_array(inputs, qgraph.cfg))

    assert graph.output.shape == (1, 2, 2)
    assert floating.output.shape == (1, 2, 2)
    assert floating.logits.shape == (1, 2, 2)
    assert fixed.output.shape == (1, 2, 2)
    assert fixed.logits.shape == (1, 2, 2)
    assert isinstance(graph.ops[-1], ReluIR if with_relu else Conv2dIR)

    emit_systemverilog(graph, qgraph.cfg, tmp_path, vector_count=2, seed=9)
    expected_logits = np.loadtxt(tmp_path / "expected_logits.txt", dtype=np.int64)
    assert expected_logits.shape == (2, 4)
    assert run_simulation(tmp_path).ok


def test_explicit_global_argmax_preserves_model_output_and_pre_argmax_logits(
    tmp_path: Path,
) -> None:
    class ConvArgmax(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 2, kernel_size=2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return torch.argmax(self.conv(inputs))

    model = ConvArgmax().eval()
    inputs = torch.tensor(
        [[[0.25, -0.5, 0.75], [1.0, -0.25, 0.5], [-0.75, 0.125, 0.625]]]
    )
    graph = parse_model(model, input_shape=(1, 3, 3))
    floating = infer_float_graph(graph, inputs.numpy())
    qgraph = quantize_graph(graph, FixedPointConfig())
    fixed = infer_quantized(qgraph, quantize_array(inputs.numpy(), qgraph.cfg))

    assert isinstance(graph.ops[-1], ArgmaxIR)
    assert graph.output.shape == ()
    assert floating.logits.shape == (2, 2, 2)
    assert floating.output.shape == ()
    assert int(floating.output) == floating.class_id == int(model(inputs))
    assert floating.activations[-1].name == graph.ops[-1].name
    assert floating.activations[-1].values.shape == ()
    assert fixed.logits.shape == (2, 2, 2)
    assert fixed.output.shape == ()
    assert int(fixed.output) == fixed.class_id
    assert fixed.activations[-1].values.shape == ()

    emit_systemverilog(graph, qgraph.cfg, tmp_path, vector_count=2, seed=10)
    assert np.loadtxt(tmp_path / "expected_logits.txt").shape == (2, 8)
    assert run_simulation(tmp_path).ok


@pytest.mark.parametrize("operation", ["relu", "argmax"])
def test_fixed_reference_preserves_multidimensional_input_shape_until_flatten(
    operation: str,
) -> None:
    class DirectArgmax(nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return torch.argmax(inputs)

    model: nn.Module = nn.Sequential(nn.ReLU())
    if operation == "argmax":
        model = DirectArgmax()
    inputs = np.asarray(
        [[-1.0, 0.25, 0.5], [0.75, -0.5, 0.125]],
        dtype=np.float32,
    )
    graph = parse_model(model.eval(), input_shape=(2, 3))
    qgraph = quantize_graph(graph, FixedPointConfig())
    result = infer_quantized(qgraph, quantize_array(inputs, qgraph.cfg))

    assert result.logits.shape == (2, 3)
    if operation == "relu":
        assert result.output.shape == (2, 3)
        assert result.activations[-1].values.shape == (2, 3)
    else:
        assert graph.output.dtype == "int64"
        assert result.output.shape == ()


def test_float32_reference_does_not_change_class_by_promoting_to_float64() -> None:
    model = nn.Sequential(nn.Linear(3, 2)).eval()
    with torch.no_grad():
        model[0].weight.copy_(
            torch.tensor([[1.0e8, 1.0, -1.0e8], [0.0, 0.0, 0.0]])
        )
        model[0].bias.copy_(torch.tensor([0.0, 0.5]))
    inputs = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    graph = parse_model(model, input_shape=(3,))
    result = infer_float_graph(graph, inputs.numpy())
    expected = model(inputs).detach().numpy()

    assert graph.input.dtype == "float32"
    assert graph.output.dtype == "float32"
    assert graph.ops[0].weight.dtype == np.float32
    assert result.logits.dtype == np.float32
    np.testing.assert_array_equal(result.output, expected)
    assert result.class_id == int(torch.argmax(model(inputs))) == 1


def test_float32_linear_reference_uses_pytorch_accumulation_semantics() -> None:
    weight = torch.tensor(
        [
            1024.0,
            16384.0,
            -0.0625,
            -1024.0,
            16384.0,
            32768.0,
            0.0009765625,
            131072.0,
        ],
        dtype=torch.float32,
    )
    inputs = torch.tensor(
        [
            0.10558542609214783,
            0.373366117477417,
            0.5779691338539124,
            0.5151948928833008,
            -0.011456655338406563,
            -0.39703086018562317,
            0.3128998279571533,
            -0.7188253998756409,
        ],
        dtype=torch.float32,
    )
    model = nn.Sequential(nn.Linear(8, 2)).eval()
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].weight[0].copy_(weight)
        tie_value = torch.nn.functional.linear(inputs, weight.reshape(1, -1)).item()
        model[0].bias.copy_(torch.tensor([0.0, tie_value]))

    graph = parse_model(model, input_shape=(8,))
    result = infer_float_graph(graph, inputs.numpy())
    expected = model(inputs).detach().numpy()

    np.testing.assert_array_equal(result.logits, expected)
    assert result.class_id == int(torch.argmax(model(inputs))) == 0


def test_float32_conv_reference_uses_pytorch_accumulation_semantics() -> None:
    weight = torch.tensor(
        [-2048.0, -65536.0, -512.0, 4194304.0],
        dtype=torch.float32,
    )
    inputs = torch.tensor(
        [
            0.2500721514225006,
            -0.24794362485408783,
            0.5054433941841125,
            0.4355789124965668,
        ],
        dtype=torch.float32,
    ).reshape(1, 1, 4)
    model = nn.Sequential(nn.Conv2d(1, 2, kernel_size=(1, 4))).eval()
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].weight[0, 0, 0].copy_(weight)
        tie_value = torch.nn.functional.conv2d(
            inputs,
            model[0].weight[:1],
            None,
        ).item()
        model[0].bias.copy_(torch.tensor([0.0, tie_value]))

    graph = parse_model(model, input_shape=(1, 1, 4))
    result = infer_float_graph(graph, inputs.numpy())
    expected = model(inputs).detach().numpy()

    np.testing.assert_array_equal(result.logits, expected)
    assert result.class_id == int(torch.argmax(model(inputs))) == 0


def test_float64_model_and_reference_stay_float64() -> None:
    model = nn.Sequential(nn.Linear(3, 2)).double().eval()
    inputs = torch.tensor([0.25, -0.5, 0.75], dtype=torch.float64)
    graph = parse_model(model, input_shape=(3,))
    result = infer_float_graph(graph, inputs.numpy())

    assert graph.input.dtype == "float64"
    assert graph.ops[0].weight.dtype == np.float64
    assert result.logits.dtype == np.float64
    np.testing.assert_allclose(
        result.output,
        model(inputs).detach().numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
