from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from torch2rtl.quant.fixed_point import FixedPointConfig
from torch2rtl.quant.reference import (
    QuantizedArgmaxIR,
    QuantizedConv2dIR,
    QuantizedFlattenIR,
    QuantizedGraph,
    QuantizedLinearIR,
    QuantizedOp,
    QuantizedReluIR,
    conv2d_fixed,
    linear_fixed,
)

PREVIEW_LIMIT = 12


def trace_payload(
    qgraph: QuantizedGraph,
    manifest: dict[str, Any],
    build_dir: Path,
    vector_index: int,
) -> dict[str, Any]:
    inputs = _load_input_vectors(build_dir / "input_vectors.txt", qgraph.input_size)
    expected = _load_expected_classes(build_dir / "expected_classes.txt")
    if vector_index < 0 or vector_index >= len(inputs):
        raise IndexError(f"vector_index {vector_index} outside 0..{len(inputs) - 1}")

    data = np.asarray(inputs[vector_index], dtype=np.int64).reshape(-1)
    steps: list[dict[str, Any]] = []
    logits = data
    class_id = 0
    expected_class = int(expected[vector_index]) if vector_index < len(expected) else None

    for op, record in zip(qgraph.ops, manifest["ops"], strict=True):
        before = logits
        if isinstance(op, QuantizedLinearIR):
            logits = linear_fixed(logits, op.weight, op.bias, qgraph.cfg)
            details = {
                "weights": array_stats(op.weight),
                "biases": array_stats(op.bias),
                "macs": op.in_features * op.out_features,
            }
        elif isinstance(op, QuantizedConv2dIR):
            logits = conv2d_fixed(logits, op, qgraph.cfg)
            details = {
                "weights": array_stats(op.weight),
                "biases": array_stats(op.bias),
                "macs": op.mac_count,
                "kernel_size": f"{op.kernel_height}x{op.kernel_width}",
                "output_shape": f"{op.out_channels}x{op.output_height}x{op.output_width}",
            }
        elif isinstance(op, QuantizedReluIR):
            logits = np.maximum(logits, 0).astype(np.int64)
            details = {"clamped_values": int(np.count_nonzero(before < 0))}
        elif isinstance(op, QuantizedFlattenIR):
            logits = logits.reshape(-1)
            details = {"reshape": "flat"}
        elif isinstance(op, QuantizedArgmaxIR):
            class_id = int(np.argmax(logits))
            details = {"winner_index": class_id, "winner_value": int(logits[class_id])}
        else:
            raise TypeError(f"Unsupported quantized op for trace: {type(op).__name__}")

        steps.append(
            {
                "block_id": record["id"],
                "kind": record["kind"],
                "label": record["kind"],
                "input": array_summary(before),
                "output": array_summary(logits),
                "details": details,
            }
        )

    return {
        "vector_index": vector_index,
        "input": array_summary(data),
        "steps": steps,
        "logits": array_summary(logits),
        "class_id": class_id,
        "expected_class": expected_class,
        "matched": expected_class is None or expected_class == class_id,
    }


def qgraph_from_manifest(manifest: dict[str, Any], build_dir: Path) -> QuantizedGraph:
    cfg = FixedPointConfig(**manifest["quant"])
    ops: list[QuantizedOp] = []
    for record in manifest["ops"]:
        if record["kind"] == "linear":
            weight = np.loadtxt(build_dir / record["weight_file"], dtype=np.int64).reshape(
                record["out_features"],
                record["in_features"],
            )
            bias = np.loadtxt(build_dir / record["bias_file"], dtype=np.int64).reshape(
                record["out_features"]
            )
            ops.append(
                QuantizedLinearIR(
                    name=record["source_name"],
                    in_features=record["in_features"],
                    out_features=record["out_features"],
                    weight=weight,
                    bias=bias,
                )
            )
        elif record["kind"] == "conv2d":
            weight = np.loadtxt(build_dir / record["weight_file"], dtype=np.int64).reshape(
                record["out_channels"],
                record["in_channels"],
                record["kernel_height"],
                record["kernel_width"],
            )
            bias = np.loadtxt(build_dir / record["bias_file"], dtype=np.int64).reshape(
                record["out_channels"]
            )
            ops.append(
                QuantizedConv2dIR(
                    name=record["source_name"],
                    in_channels=record["in_channels"],
                    out_channels=record["out_channels"],
                    input_height=record["input_height"],
                    input_width=record["input_width"],
                    output_height=record["output_height"],
                    output_width=record["output_width"],
                    kernel_height=record["kernel_height"],
                    kernel_width=record["kernel_width"],
                    stride=tuple(record["stride"]),
                    padding=tuple(record["padding"]),
                    weight=weight,
                    bias=bias,
                )
            )
        elif record["kind"] == "relu":
            ops.append(QuantizedReluIR(name=record["source_name"]))
        elif record["kind"] == "flatten":
            ops.append(QuantizedFlattenIR(name=record["source_name"]))
        elif record["kind"] == "argmax":
            ops.append(QuantizedArgmaxIR(name=record["source_name"]))
        else:
            raise ValueError(f"Unsupported manifest op kind: {record['kind']}")
    return QuantizedGraph(
        input_shape=tuple(manifest["graph"]["input_shape"]),
        ops=tuple(ops),
        cfg=cfg,
    )


def array_summary(values: np.ndarray) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.int64).reshape(-1)
    return {
        "size": int(data.size),
        "preview": [int(value) for value in data[:PREVIEW_LIMIT]],
        "min": int(data.min()) if data.size else 0,
        "max": int(data.max()) if data.size else 0,
        "mean": round(float(data.mean()), 3) if data.size else 0.0,
        "nonzero": int(np.count_nonzero(data)),
    }


def array_stats(values: np.ndarray) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.int64).reshape(-1)
    return {
        "count": int(data.size),
        "min": int(data.min()) if data.size else 0,
        "max": int(data.max()) if data.size else 0,
        "zero_count": int(data.size - np.count_nonzero(data)),
    }


def _load_input_vectors(path: Path, input_size: int) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.int64)
    data = np.asarray(values, dtype=np.int64)
    if data.ndim == 0:
        return data.reshape(1, 1)
    if data.ndim == 1:
        if input_size == 1:
            return data.reshape(-1, 1)
        return data.reshape(1, input_size)
    return data


def _load_expected_classes(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.int64)
    return np.asarray(values, dtype=np.int64).reshape(-1)
