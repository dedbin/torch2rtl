from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from torch2rtl.ir.graph import GraphIR
from torch2rtl.ir.ops import ArgmaxIR, Conv2dIR, FlattenIR, LinearIR, ReluIR
from torch2rtl.quant.fixed_point import FixedPointConfig, quantize_array, saturate_int


@dataclass(frozen=True)
class QuantizedLinearIR:
    name: str
    in_features: int
    out_features: int
    weight: np.ndarray
    bias: np.ndarray


@dataclass(frozen=True)
class QuantizedConv2dIR:
    name: str
    in_channels: int
    out_channels: int
    input_height: int
    input_width: int
    output_height: int
    output_width: int
    kernel_height: int
    kernel_width: int
    stride: tuple[int, int]
    padding: tuple[int, int]
    weight: np.ndarray
    bias: np.ndarray

    @property
    def input_size(self) -> int:
        return self.in_channels * self.input_height * self.input_width

    @property
    def output_size(self) -> int:
        return self.out_channels * self.output_height * self.output_width

    @property
    def parameter_count(self) -> int:
        return int(self.weight.size + self.bias.size)

    @property
    def mac_count(self) -> int:
        return (
            self.out_channels
            * self.output_height
            * self.output_width
            * self.in_channels
            * self.kernel_height
            * self.kernel_width
        )


@dataclass(frozen=True)
class QuantizedReluIR:
    name: str


@dataclass(frozen=True)
class QuantizedFlattenIR:
    name: str


@dataclass(frozen=True)
class QuantizedArgmaxIR:
    name: str


QuantizedOp: TypeAlias = (
    QuantizedLinearIR
    | QuantizedConv2dIR
    | QuantizedReluIR
    | QuantizedFlattenIR
    | QuantizedArgmaxIR
)


@dataclass(frozen=True)
class QuantizedGraph:
    input_shape: tuple[int, ...]
    ops: tuple[QuantizedOp, ...]
    cfg: FixedPointConfig

    @property
    def input_size(self) -> int:
        total = 1
        for dim in self.input_shape:
            total *= dim
        return total


@dataclass(frozen=True)
class ReferenceResult:
    logits: np.ndarray
    class_id: int


@dataclass(frozen=True)
class FloatReferenceResult:
    logits: np.ndarray
    class_id: int


def quantize_graph(graph: GraphIR, cfg: FixedPointConfig) -> QuantizedGraph:
    qops: list[QuantizedOp] = []
    for op in graph.ops:
        if isinstance(op, LinearIR):
            bias = op.bias if op.bias is not None else np.zeros(op.out_features)
            qops.append(
                QuantizedLinearIR(
                    name=op.name,
                    in_features=op.in_features,
                    out_features=op.out_features,
                    weight=quantize_array(op.weight, cfg),
                    bias=quantize_array(bias, cfg),
                )
            )
        elif isinstance(op, Conv2dIR):
            bias = op.bias if op.bias is not None else np.zeros(op.out_channels)
            qops.append(
                QuantizedConv2dIR(
                    name=op.name,
                    in_channels=op.in_channels,
                    out_channels=op.out_channels,
                    input_height=op.input_height,
                    input_width=op.input_width,
                    output_height=op.output_height,
                    output_width=op.output_width,
                    kernel_height=op.kernel_height,
                    kernel_width=op.kernel_width,
                    stride=op.stride,
                    padding=op.padding,
                    weight=quantize_array(op.weight, cfg),
                    bias=quantize_array(bias, cfg),
                )
            )
        elif isinstance(op, ReluIR):
            qops.append(QuantizedReluIR(name=op.name))
        elif isinstance(op, FlattenIR):
            qops.append(QuantizedFlattenIR(name=op.name))
        elif isinstance(op, ArgmaxIR):
            qops.append(QuantizedArgmaxIR(name=op.name))
        else:
            raise TypeError(f"Unsupported IR op for quantization: {type(op).__name__}")
    return QuantizedGraph(input_shape=graph.input.shape, ops=tuple(qops), cfg=cfg)


def linear_fixed(
    inputs: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    cfg: FixedPointConfig,
) -> np.ndarray:
    x = np.asarray(inputs, dtype=np.int64).reshape(-1)
    w = np.asarray(weight, dtype=np.int64)
    b = np.asarray(bias, dtype=np.int64)
    if w.shape[1] != x.shape[0]:
        raise ValueError(f"Linear input mismatch: weight {w.shape}, input {x.shape}")
    outputs: list[int] = []
    for out_idx in range(w.shape[0]):
        acc = int(b[out_idx]) << cfg.frac_bits
        for in_idx in range(w.shape[1]):
            acc += int(x[in_idx]) * int(w[out_idx, in_idx])
        shifted = acc >> cfg.frac_bits
        outputs.append(saturate_int(shifted, cfg.bits))
    return np.asarray(outputs, dtype=np.int64)


def conv2d_fixed(
    inputs: np.ndarray,
    op: QuantizedConv2dIR,
    cfg: FixedPointConfig,
) -> np.ndarray:
    data = np.asarray(inputs, dtype=np.int64).reshape(
        op.in_channels,
        op.input_height,
        op.input_width,
    )
    weight = np.asarray(op.weight, dtype=np.int64).reshape(
        op.out_channels,
        op.in_channels,
        op.kernel_height,
        op.kernel_width,
    )
    bias = np.asarray(op.bias, dtype=np.int64).reshape(op.out_channels)
    output = np.zeros(
        (op.out_channels, op.output_height, op.output_width),
        dtype=np.int64,
    )
    for out_channel in range(op.out_channels):
        for out_y in range(op.output_height):
            for out_x in range(op.output_width):
                acc = int(bias[out_channel]) << cfg.frac_bits
                for in_channel in range(op.in_channels):
                    for kernel_y in range(op.kernel_height):
                        in_y = out_y * op.stride[0] + kernel_y - op.padding[0]
                        if in_y < 0 or in_y >= op.input_height:
                            continue
                        for kernel_x in range(op.kernel_width):
                            in_x = out_x * op.stride[1] + kernel_x - op.padding[1]
                            if in_x < 0 or in_x >= op.input_width:
                                continue
                            acc += int(data[in_channel, in_y, in_x]) * int(
                                weight[out_channel, in_channel, kernel_y, kernel_x]
                            )
                output[out_channel, out_y, out_x] = saturate_int(
                    acc >> cfg.frac_bits,
                    cfg.bits,
                )
    return output


def infer_float_graph(graph: GraphIR, input_values: np.ndarray) -> FloatReferenceResult:
    logits = np.asarray(input_values, dtype=np.float64)
    class_id = 0
    for op in graph.ops:
        if isinstance(op, LinearIR):
            data = logits.reshape(-1)
            bias = op.bias if op.bias is not None else np.zeros(op.out_features)
            logits = op.weight @ data + bias
        elif isinstance(op, Conv2dIR):
            logits = _conv2d_float(logits, op)
        elif isinstance(op, ReluIR):
            logits = np.maximum(logits, 0.0)
        elif isinstance(op, FlattenIR):
            logits = logits.reshape(-1)
        elif isinstance(op, ArgmaxIR):
            class_id = int(np.argmax(logits))
        else:
            raise TypeError(f"Unsupported IR op for float reference: {type(op).__name__}")
    return FloatReferenceResult(logits=np.asarray(logits, dtype=np.float64), class_id=class_id)


def _conv2d_float(inputs: np.ndarray, op: Conv2dIR) -> np.ndarray:
    data = np.asarray(inputs, dtype=np.float64).reshape(
        op.in_channels,
        op.input_height,
        op.input_width,
    )
    bias = op.bias if op.bias is not None else np.zeros(op.out_channels)
    output = np.zeros(
        (op.out_channels, op.output_height, op.output_width),
        dtype=np.float64,
    )
    for out_channel in range(op.out_channels):
        for out_y in range(op.output_height):
            for out_x in range(op.output_width):
                acc = float(bias[out_channel])
                for in_channel in range(op.in_channels):
                    for kernel_y in range(op.kernel_height):
                        in_y = out_y * op.stride[0] + kernel_y - op.padding[0]
                        if in_y < 0 or in_y >= op.input_height:
                            continue
                        for kernel_x in range(op.kernel_width):
                            in_x = out_x * op.stride[1] + kernel_x - op.padding[1]
                            if in_x < 0 or in_x >= op.input_width:
                                continue
                            acc += (
                                float(data[in_channel, in_y, in_x])
                                * float(op.weight[out_channel, in_channel, kernel_y, kernel_x])
                            )
                output[out_channel, out_y, out_x] = acc
    return output


def infer_quantized(qgraph: QuantizedGraph, input_values: np.ndarray) -> ReferenceResult:
    data = np.asarray(input_values, dtype=np.int64).reshape(-1)
    logits = data
    class_id = 0
    for op in qgraph.ops:
        if isinstance(op, QuantizedLinearIR):
            logits = linear_fixed(logits, op.weight, op.bias, qgraph.cfg)
        elif isinstance(op, QuantizedConv2dIR):
            logits = conv2d_fixed(logits, op, qgraph.cfg)
        elif isinstance(op, QuantizedReluIR):
            logits = np.maximum(logits, 0).astype(np.int64)
        elif isinstance(op, QuantizedFlattenIR):
            logits = logits.reshape(-1)
        elif isinstance(op, QuantizedArgmaxIR):
            class_id = int(np.argmax(logits))
        else:
            raise TypeError(f"Unsupported quantized op: {type(op).__name__}")
    return ReferenceResult(logits=np.asarray(logits, dtype=np.int64), class_id=class_id)
