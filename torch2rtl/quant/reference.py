from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from torch2rtl.ir.graph import GraphIR
from torch2rtl.ir.ops import ArgmaxIR, FlattenIR, LinearIR, ReluIR
from torch2rtl.quant.fixed_point import FixedPointConfig, quantize_array, saturate_int


@dataclass(frozen=True)
class QuantizedLinearIR:
    name: str
    in_features: int
    out_features: int
    weight: np.ndarray
    bias: np.ndarray


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
    QuantizedLinearIR | QuantizedReluIR | QuantizedFlattenIR | QuantizedArgmaxIR
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


def infer_quantized(qgraph: QuantizedGraph, input_values: np.ndarray) -> ReferenceResult:
    data = np.asarray(input_values, dtype=np.int64).reshape(-1)
    logits = data
    class_id = 0
    for op in qgraph.ops:
        if isinstance(op, QuantizedLinearIR):
            logits = linear_fixed(logits, op.weight, op.bias, qgraph.cfg)
        elif isinstance(op, QuantizedReluIR):
            logits = np.maximum(logits, 0).astype(np.int64)
        elif isinstance(op, QuantizedFlattenIR):
            logits = logits.reshape(-1)
        elif isinstance(op, QuantizedArgmaxIR):
            class_id = int(np.argmax(logits))
        else:
            raise TypeError(f"Unsupported quantized op: {type(op).__name__}")
    return ReferenceResult(logits=np.asarray(logits, dtype=np.int64), class_id=class_id)
