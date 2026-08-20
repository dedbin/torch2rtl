from __future__ import annotations

import importlib.util
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import numpy as np
import pytest
import torch
import torch.nn as nn

from torch2rtl.frontend.pytorch_fx import load_model_from_file, parse_model
from torch2rtl.ir.graph import GraphIR
from torch2rtl.ir.tensor import TensorIR


def _serialize_graph(graph: GraphIR) -> dict[str, object]:
    """Readable, exact public GraphIR snapshot without repr- or address-data."""
    return {
        "input": _serialize_value(graph.input),
        "output": _serialize_value(graph.output),
        "ops": [_serialize_value(op) for op in graph.ops],
        "metadata": _serialize_value(graph.metadata),
    }


def _serialize_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "dtype": str(contiguous.dtype),
            "shape": list(contiguous.shape),
            "values": contiguous.tolist(),
            "c_order_bytes": contiguous.tobytes(order="C").hex(),
        }
    if isinstance(value, TensorIR):
        return {
            "name": value.name,
            "shape": list(value.shape),
            "dtype": value.dtype,
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": type(value).__name__,
            **{
                field.name: _serialize_value(getattr(value, field.name))
                for field in fields(value)
            },
        }
    if isinstance(value, Mapping):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serialize_value(item) for item in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError(f"Unsupported characterization value: {type(value).__name__}")


def _tensor(name: str, shape: tuple[int, ...], dtype: str) -> dict[str, object]:
    return {"name": name, "shape": list(shape), "dtype": dtype}


def _array(
    values: object,
    *,
    dtype: np.dtype[Any] | type[np.floating[Any]],
    shape: tuple[int, ...],
) -> dict[str, object]:
    array = np.asarray(values, dtype=dtype).reshape(shape)
    return {
        "dtype": str(array.dtype),
        "shape": list(shape),
        "values": array.tolist(),
        "c_order_bytes": array.tobytes(order="C").hex(),
    }


def _linear_op(
    *,
    name: str,
    input_tensor: dict[str, object],
    output_tensor: dict[str, object],
    weight: object,
    bias: object | None,
    dtype: np.dtype[Any] | type[np.floating[Any]],
    in_features: int,
    out_features: int,
) -> dict[str, object]:
    return {
        "type": "LinearIR",
        "name": name,
        "input": input_tensor,
        "output": output_tensor,
        "in_features": in_features,
        "out_features": out_features,
        "weight": _array(
            weight,
            dtype=dtype,
            shape=(out_features, in_features),
        ),
        "bias": (
            None
            if bias is None
            else _array(bias, dtype=dtype, shape=(out_features,))
        ),
    }


def _conv_op(
    *,
    name: str,
    input_tensor: dict[str, object],
    output_tensor: dict[str, object],
    weight: object,
    bias: object | None,
    dtype: np.dtype[Any] | type[np.floating[Any]],
    in_channels: int,
    out_channels: int,
    input_hw: tuple[int, int],
    output_hw: tuple[int, int],
    kernel: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> dict[str, object]:
    return {
        "type": "Conv2dIR",
        "name": name,
        "input": input_tensor,
        "output": output_tensor,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "input_height": input_hw[0],
        "input_width": input_hw[1],
        "output_height": output_hw[0],
        "output_width": output_hw[1],
        "kernel_height": kernel[0],
        "kernel_width": kernel[1],
        "stride": list(stride),
        "padding": list(padding),
        "weight": _array(
            weight,
            dtype=dtype,
            shape=(out_channels, in_channels, *kernel),
        ),
        "bias": (
            None
            if bias is None
            else _array(bias, dtype=dtype, shape=(out_channels,))
        ),
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("with_bias", [False, True])
def test_canonical_linear_bias_and_dtype_corpus(
    dtype: torch.dtype,
    with_bias: bool,
) -> None:
    model = nn.Sequential(nn.Linear(3, 2, bias=with_bias, dtype=dtype)).eval()
    weight = [[0.25, -0.5, 1.0], [1.5, 0.0, -0.25]]
    bias = [0.125, -0.75] if with_bias else None
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(weight, dtype=dtype))
        if model[0].bias is not None:
            model[0].bias.copy_(torch.tensor(bias, dtype=dtype))

    graph = parse_model(model, input_shape=(3,), input_dtype=dtype)
    dtype_name = "float64" if dtype is torch.float64 else "float32"
    numpy_dtype = np.float64 if dtype is torch.float64 else np.float32
    input_tensor = _tensor("input", (3,), dtype_name)
    output_tensor = _tensor("_0_out", (2,), dtype_name)

    assert _serialize_graph(graph) == {
        "input": input_tensor,
        "output": output_tensor,
        "ops": [
            _linear_op(
                name="_0",
                input_tensor=input_tensor,
                output_tensor=output_tensor,
                weight=weight,
                bias=bias,
                dtype=numpy_dtype,
                in_features=3,
                out_features=2,
            )
        ],
        "metadata": {"source": "Sequential"},
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("with_bias", [False, True])
def test_canonical_conv_bias_and_dtype_corpus(
    dtype: torch.dtype,
    with_bias: bool,
) -> None:
    model = nn.Sequential(
        nn.Conv2d(
            1,
            2,
            kernel_size=(2, 2),
            stride=(1, 2),
            padding=(1, 0),
            bias=with_bias,
            dtype=dtype,
        )
    ).eval()
    weight = [
        [[[0.25, -0.5], [0.75, 1.0]]],
        [[[-1.0, 0.5], [0.125, -0.25]]],
    ]
    bias = [0.5, -0.75] if with_bias else None
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(weight, dtype=dtype))
        if model[0].bias is not None:
            model[0].bias.copy_(torch.tensor(bias, dtype=dtype))

    graph = parse_model(model, input_shape=(1, 3, 4), input_dtype=dtype)
    dtype_name = "float64" if dtype is torch.float64 else "float32"
    numpy_dtype = np.float64 if dtype is torch.float64 else np.float32
    input_tensor = _tensor("input", (1, 3, 4), dtype_name)
    output_tensor = _tensor("_0_out", (2, 4, 2), dtype_name)

    assert _serialize_graph(graph) == {
        "input": input_tensor,
        "output": output_tensor,
        "ops": [
            _conv_op(
                name="_0",
                input_tensor=input_tensor,
                output_tensor=output_tensor,
                weight=weight,
                bias=bias,
                dtype=numpy_dtype,
                in_channels=1,
                out_channels=2,
                input_hw=(3, 4),
                output_hw=(4, 2),
                kernel=(2, 2),
                stride=(1, 2),
                padding=(1, 0),
            )
        ],
        "metadata": {"source": "Sequential"},
    }


def test_canonical_relu_and_negative_full_flatten_corpus() -> None:
    model = nn.Sequential(nn.ReLU(), nn.Flatten(start_dim=-2, end_dim=-1)).eval()
    graph = parse_model(model, input_shape=(2, 3))
    input_tensor = _tensor("input", (2, 3), "float32")
    relu_output = _tensor("_0_out", (2, 3), "float32")
    flatten_output = _tensor("_1_out", (6,), "float32")

    assert _serialize_graph(graph) == {
        "input": input_tensor,
        "output": flatten_output,
        "ops": [
            {
                "type": "ReluIR",
                "name": "_0",
                "input": input_tensor,
                "output": relu_output,
            },
            {
                "type": "FlattenIR",
                "name": "_1",
                "input": relu_output,
                "output": flatten_output,
            },
        ],
        "metadata": {"source": "Sequential"},
    }


class _TorchArgmax(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.argmax(inputs)


class _TensorArgmax(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.argmax()


@pytest.mark.parametrize(
    "model_type",
    [_TorchArgmax, _TensorArgmax],
    ids=["torch", "tensor"],
)
def test_canonical_explicit_argmax_corpus(
    model_type: type[nn.Module],
) -> None:
    model = model_type()

    graph = parse_model(model, input_shape=(2, 3))
    input_tensor = _tensor("input", (2, 3), "float32")
    output_tensor = _tensor("argmax", (), "int64")

    assert _serialize_graph(graph) == {
        "input": input_tensor,
        "output": output_tensor,
        "ops": [
            {
                "type": "ArgmaxIR",
                "name": "argmax",
                "input": input_tensor,
                "output": output_tensor,
            }
        ],
        "metadata": {"source": model_type.__name__},
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("conv_bias", [False, True])
@pytest.mark.parametrize("bn_affine", [False, True])
def test_canonical_conv_batchnorm_matrix(
    dtype: torch.dtype,
    conv_bias: bool,
    bn_affine: bool,
) -> None:
    conv = nn.Conv2d(1, 1, 1, bias=conv_bias, dtype=dtype)
    batchnorm = nn.BatchNorm2d(1, eps=1.0, affine=bn_affine, dtype=dtype)
    with torch.no_grad():
        conv.weight.fill_(2.0)
        if conv.bias is not None:
            conv.bias.fill_(3.0)
        assert batchnorm.running_mean is not None
        assert batchnorm.running_var is not None
        batchnorm.running_mean.fill_(1.0)
        batchnorm.running_var.fill_(3.0)
        if batchnorm.weight is not None:
            batchnorm.weight.fill_(4.0)
        if batchnorm.bias is not None:
            batchnorm.bias.fill_(5.0)
    model = nn.Sequential(conv, batchnorm).eval()

    graph = parse_model(model, input_shape=(1, 2, 3), input_dtype=dtype)
    dtype_name = "float64" if dtype is torch.float64 else "float32"
    numpy_dtype = np.float64 if dtype is torch.float64 else np.float32
    input_tensor = _tensor("input", (1, 2, 3), dtype_name)
    output_tensor = _tensor("_0_out", (1, 2, 3), dtype_name)
    expected_weight = 4.0 if bn_affine else 1.0
    if bn_affine:
        expected_bias = 9.0 if conv_bias else 3.0
    else:
        expected_bias = 1.0 if conv_bias else -0.5

    assert _serialize_graph(graph) == {
        "input": input_tensor,
        "output": output_tensor,
        "ops": [
            _conv_op(
                name="_0",
                input_tensor=input_tensor,
                output_tensor=output_tensor,
                weight=[[[[expected_weight]]]],
                bias=[expected_bias],
                dtype=numpy_dtype,
                in_channels=1,
                out_channels=1,
                input_hw=(2, 3),
                output_hw=(2, 3),
                kernel=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            )
        ],
        "metadata": {
            "source": "Sequential",
            "input_adapter": {"kind": "singleton_batch_n1"},
            "transformations": [
                {
                    "kind": "conv2d_batchnorm2d_fusion",
                    "conv_node": "_0",
                    "conv_target": "0",
                    "batchnorm_node": "_1",
                    "batchnorm_target": "1",
                    "fused_target": "_torch2rtl_fused_conv_bn_0",
                }
            ],
        },
    }
    with torch.no_grad():
        assert tuple(model(torch.zeros(1, 1, 2, 3, dtype=dtype)).shape) == (1, 1, 2, 3)
    assert graph.input.shape == (1, 2, 3)
    assert graph.output.shape == (1, 2, 3)


class _TwoFusionPairs(nn.Module):
    def __init__(self, *, reuse: bool) -> None:
        super().__init__()
        self.first_conv = nn.Conv2d(1, 1, 1)
        self.first_bn = nn.BatchNorm2d(1)
        if reuse:
            self.second_conv = self.first_conv
            self.second_bn = self.first_bn
        else:
            self.second_conv = nn.Conv2d(1, 1, 1)
            self.second_bn = nn.BatchNorm2d(1)
        with torch.no_grad():
            self.first_conv.weight.fill_(1.0)
            self.first_conv.bias.zero_()
            self.first_bn.running_mean.zero_()
            self.first_bn.running_var.fill_(1.0)
            self.first_bn.weight.fill_(1.0)
            self.first_bn.bias.zero_()
            if not reuse:
                self.second_conv.weight.fill_(2.0)
                self.second_conv.bias.fill_(0.25)
                self.second_bn.running_mean.fill_(0.5)
                self.second_bn.running_var.fill_(1.0)
                self.second_bn.weight.fill_(0.5)
                self.second_bn.bias.fill_(0.75)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        first = self.first_bn(self.first_conv(inputs))
        return self.second_bn(self.second_conv(first))


@pytest.mark.parametrize("reuse", [False, True], ids=["distinct", "reused"])
def test_canonical_multiple_and_reused_fusion_pairs(reuse: bool) -> None:
    graph = parse_model(_TwoFusionPairs(reuse=reuse).eval(), input_shape=(1, 2, 2))
    snapshot = _serialize_graph(graph)

    expected_names = (
        ["first_conv", "first_conv_1"]
        if reuse
        else ["first_conv", "second_conv"]
    )
    assert [op["name"] for op in snapshot["ops"]] == expected_names  # type: ignore[index]
    assert snapshot["metadata"] == {
        "source": "_TwoFusionPairs",
        "input_adapter": {"kind": "singleton_batch_n1"},
        "transformations": [
            {
                "kind": "conv2d_batchnorm2d_fusion",
                "conv_node": "first_conv",
                "conv_target": "first_conv",
                "batchnorm_node": "first_bn",
                "batchnorm_target": "first_bn",
                "fused_target": "_torch2rtl_fused_conv_bn_0",
            },
                {
                    "kind": "conv2d_batchnorm2d_fusion",
                    "conv_node": "first_conv_1" if reuse else "second_conv",
                    "conv_target": "first_conv" if reuse else "second_conv",
                    "batchnorm_node": "first_bn_1" if reuse else "second_bn",
                "batchnorm_target": "first_bn" if reuse else "second_bn",
                "fused_target": "_torch2rtl_fused_conv_bn_1",
            },
        ],
    }
    first, second = graph.ops
    np.testing.assert_array_equal(first.weight, [[[[0.9999949932098389]]]])
    np.testing.assert_array_equal(first.bias, [0.0])
    if reuse:
        np.testing.assert_array_equal(second.weight, first.weight)
        np.testing.assert_array_equal(second.bias, first.bias)
    else:
        np.testing.assert_array_equal(second.weight, [[[[0.9999949932098389]]]])
        np.testing.assert_array_equal(second.bias, [0.6250005960464478])


@pytest.mark.parametrize(
    ("example", "shape", "expected_types"),
    [
        (
            "tiny_conv",
            (1, 3, 3),
            ["Conv2dIR", "ReluIR", "FlattenIR", "LinearIR"],
        ),
        (
            "grid_classifier",
            (4, 4),
            [
                "FlattenIR",
                "LinearIR",
                "ReluIR",
                "LinearIR",
                "ReluIR",
                "LinearIR",
            ],
        ),
        (
            "image_cnn",
            (1, 4, 4),
            ["Conv2dIR", "ReluIR", "FlattenIR", "LinearIR"],
        ),
    ],
)
def test_checked_in_examples_are_in_canonical_public_corpus(
    example: str,
    shape: tuple[int, ...],
    expected_types: list[str],
) -> None:
    path = Path("examples") / example / "model.py"
    model = load_model_from_file(path)
    assert isinstance(model, nn.Module)
    if example == "image_cnn":
        with torch.no_grad():
            conv = model.net[0]
            linear = model.net[3]
            conv.weight.copy_(torch.arange(36, dtype=torch.float32).reshape(4, 1, 3, 3) / 32)
            conv.bias.copy_(torch.tensor([-0.25, 0.0, 0.25, 0.5]))
            linear.weight.copy_(
                torch.arange(256, dtype=torch.float32).reshape(4, 64) / 512 - 0.25
            )
            linear.bias.copy_(torch.tensor([0.5, 0.25, 0.0, -0.25]))

    graph = parse_model(model, input_shape=shape)
    first = _serialize_graph(graph)
    second = _serialize_graph(parse_model(model, input_shape=shape))

    assert first == second
    assert first["input"] == _tensor("input", shape, "float32")
    assert first["metadata"] == {"source": type(model).__name__}
    assert [op["type"] for op in first["ops"]] == expected_types  # type: ignore[index]
    expected_skeletons = {
        "tiny_conv": [
            ("Conv2dIR", "net_0", (1, 3, 3), (1, 2, 2), (2, 2), (1, 1), (0, 0)),
            ("ReluIR", "net_1", (1, 2, 2), (1, 2, 2)),
            ("FlattenIR", "net_2", (1, 2, 2), (4,)),
            ("LinearIR", "net_3", (4,), (4,), 4, 4),
        ],
        "grid_classifier": [
            ("FlattenIR", "net_0", (4, 4), (16,)),
            ("LinearIR", "net_1", (16,), (8,), 16, 8),
            ("ReluIR", "net_2", (8,), (8,)),
            ("LinearIR", "net_3", (8,), (8,), 8, 8),
            ("ReluIR", "net_4", (8,), (8,)),
            ("LinearIR", "net_5", (8,), (4,), 8, 4),
        ],
        "image_cnn": [
            ("Conv2dIR", "net_0", (1, 4, 4), (4, 4, 4), (3, 3), (1, 1), (1, 1)),
            ("ReluIR", "net_1", (4, 4, 4), (4, 4, 4)),
            ("FlattenIR", "net_2", (4, 4, 4), (64,)),
            ("LinearIR", "net_3", (64,), (4,), 64, 4),
        ],
    }
    skeleton: list[tuple[object, ...]] = []
    for operation in graph.ops:
        common: tuple[object, ...] = (
            type(operation).__name__,
            operation.name,
            operation.input.shape,
            operation.output.shape,
        )
        if type(operation).__name__ == "Conv2dIR":
            skeleton.append(
                (*common, (operation.kernel_height, operation.kernel_width), operation.stride, operation.padding)
            )
        elif type(operation).__name__ == "LinearIR":
            skeleton.append((*common, operation.in_features, operation.out_features))
        else:
            skeleton.append(common)
    assert skeleton == expected_skeletons[example]

    source_layers = [
        layer
        for layer in model.modules()
        if type(layer) in (nn.Conv2d, nn.Linear)
    ]
    graph_layers = [
        operation
        for operation in graph.ops
        if type(operation).__name__ in ("Conv2dIR", "LinearIR")
    ]
    assert len(source_layers) == len(graph_layers)
    for source_layer, graph_layer in zip(source_layers, graph_layers, strict=True):
        np.testing.assert_array_equal(
            graph_layer.weight,
            source_layer.weight.detach().cpu().numpy(),
        )
        if source_layer.bias is None:
            assert graph_layer.bias is None
        else:
            np.testing.assert_array_equal(
                graph_layer.bias,
                source_layer.bias.detach().cpu().numpy(),
            )
    for op in first["ops"]:  # type: ignore[assignment]
        if "weight" in op:
            assert op["weight"]["values"]  # type: ignore[index]
            assert op["weight"]["c_order_bytes"]  # type: ignore[index]
        if op.get("bias") is not None:
            assert op["bias"]["values"]  # type: ignore[index]
            assert op["bias"]["c_order_bytes"]  # type: ignore[index]


def test_serializer_records_array_layout_values_and_bias_none_separately() -> None:
    model = nn.Sequential(nn.Linear(2, 1, bias=False)).eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[0.25, -0.5]]))

    snapshot = _serialize_graph(parse_model(model, input_shape=(2,)))
    operation = snapshot["ops"][0]  # type: ignore[index]

    assert operation["weight"] == {  # type: ignore[index]
        "dtype": "float32",
        "shape": [1, 2],
        "values": [[0.25, -0.5]],
        "c_order_bytes": "0000803e000000bf",
    }
    assert operation["bias"] is None  # type: ignore[index]
