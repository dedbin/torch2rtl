from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from torch2rtl.quant.fixed_point import FixedPointConfig, saturate_int
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

# Visualization payloads are embedded into a self-contained HTML document.  These
# caps keep that document bounded for production-sized models while preserving
# complete data for the small teaching examples the visualizer is designed for.
FULL_VALUES_LIMIT = 256
HARDWARE_PARAMETER_VALUE_LIMIT = 2_048
HARDWARE_OUTPUT_ENTRY_LIMIT = 32
HARDWARE_TAPS_PER_ENTRY_LIMIT = 64
TRACE_COLLECTION_LIMIT = 32


def trace_payload(
    qgraph: QuantizedGraph,
    manifest: dict[str, Any],
    build_dir: Path,
    vector_index: int,
) -> dict[str, Any]:
    inputs = _load_input_vectors(build_dir / "input_vectors.txt", qgraph.input_size)
    expected = _load_expected_classes(build_dir / "expected_classes.txt")
    return _trace_vector_payload(qgraph, manifest, inputs, expected, vector_index)


def trace_collection_payload(
    qgraph: QuantizedGraph,
    manifest: dict[str, Any],
    build_dir: Path,
    selected_index: int,
    limit: int = TRACE_COLLECTION_LIMIT,
) -> dict[str, Any]:
    """Return a bounded set of vector traces, always including the selected one."""
    if limit <= 0:
        raise ValueError("trace collection limit must be positive")

    inputs = _load_input_vectors(build_dir / "input_vectors.txt", qgraph.input_size)
    expected = _load_expected_classes(build_dir / "expected_classes.txt")
    _validate_vector_index(inputs, selected_index)

    total_count = len(inputs)
    included_indices = list(range(min(total_count, limit)))
    if selected_index not in included_indices:
        included_indices[-1] = selected_index

    traces = [
        _trace_vector_payload(qgraph, manifest, inputs, expected, index)
        for index in included_indices
    ]
    return {
        "selected_index": selected_index,
        "total_count": total_count,
        "included_count": len(traces),
        "limit": limit,
        "truncated": len(traces) < total_count,
        "included_indices": included_indices,
        "traces": traces,
    }


def _trace_vector_payload(
    qgraph: QuantizedGraph,
    manifest: dict[str, Any],
    inputs: np.ndarray,
    expected: np.ndarray,
    vector_index: int,
) -> dict[str, Any]:
    _validate_vector_index(inputs, vector_index)

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
            hardware = _linear_hardware_payload(before, op, qgraph.cfg)
        elif isinstance(op, QuantizedConv2dIR):
            logits = conv2d_fixed(logits, op, qgraph.cfg)
            details = {
                "weights": array_stats(op.weight),
                "biases": array_stats(op.bias),
                "macs": op.mac_count,
                "kernel_size": f"{op.kernel_height}x{op.kernel_width}",
                "output_shape": f"{op.out_channels}x{op.output_height}x{op.output_width}",
            }
            hardware = _conv2d_hardware_payload(before, op, qgraph.cfg)
        elif isinstance(op, QuantizedReluIR):
            logits = np.maximum(logits, 0).astype(np.int64)
            details = {"clamped_values": int(np.count_nonzero(before < 0))}
            hardware = None
        elif isinstance(op, QuantizedFlattenIR):
            logits = logits.reshape(-1)
            details = {"reshape": "flat"}
            hardware = None
        elif isinstance(op, QuantizedArgmaxIR):
            class_id = int(np.argmax(logits))
            details = {"winner_index": class_id, "winner_value": int(logits[class_id])}
            hardware = None
        else:
            raise TypeError(f"Unsupported quantized op for trace: {type(op).__name__}")

        step = {
            "block_id": record["id"],
            "kind": record["kind"],
            "label": record["kind"],
            "input": array_summary(before),
            "output": array_summary(logits),
            "details": details,
        }
        if hardware is not None:
            step["hardware"] = hardware
        steps.append(step)

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
    source = np.asarray(values, dtype=np.int64)
    data = source.reshape(-1)
    summary = {
        "shape": [int(dim) for dim in source.shape],
        "size": int(data.size),
        "preview": [int(value) for value in data[:PREVIEW_LIMIT]],
        "min": int(data.min()) if data.size else 0,
        "max": int(data.max()) if data.size else 0,
        "mean": round(float(data.mean()), 3) if data.size else 0.0,
        "nonzero": int(np.count_nonzero(data)),
    }
    if data.size <= FULL_VALUES_LIMIT:
        summary["values"] = [int(value) for value in data]
    return summary


def array_stats(values: np.ndarray) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.int64).reshape(-1)
    return {
        "count": int(data.size),
        "min": int(data.min()) if data.size else 0,
        "max": int(data.max()) if data.size else 0,
        "zero_count": int(data.size - np.count_nonzero(data)),
    }


def _linear_hardware_payload(
    inputs: np.ndarray,
    op: QuantizedLinearIR,
    cfg: FixedPointConfig,
) -> dict[str, Any]:
    data = np.asarray(inputs, dtype=np.int64).reshape(-1)
    weight = np.asarray(op.weight, dtype=np.int64).reshape(
        op.out_features,
        op.in_features,
    )
    bias = np.asarray(op.bias, dtype=np.int64).reshape(op.out_features)
    output_entries: list[dict[str, Any]] = []

    for out_idx in range(min(op.out_features, HARDWARE_OUTPUT_ENTRY_LIMIT)):
        bias_raw = int(bias[out_idx])
        acc = bias_raw << cfg.frac_bits
        taps: list[dict[str, Any]] = []
        for in_idx in range(op.in_features):
            input_raw = int(data[in_idx])
            weight_raw = int(weight[out_idx, in_idx])
            product_raw = input_raw * weight_raw
            acc += product_raw
            if len(taps) < HARDWARE_TAPS_PER_ENTRY_LIMIT:
                taps.append(
                    {
                        "tap_index": in_idx,
                        "input_index": [in_idx],
                        "weight_index": [out_idx, in_idx],
                        "input_raw": input_raw,
                        "weight_raw": weight_raw,
                        "product_raw": product_raw,
                        "accumulator_raw": acc,
                    }
                )
        output_entries.append(
            _output_entry_payload(
                index=[out_idx],
                flat_index=out_idx,
                bias_raw=bias_raw,
                accumulator_raw=acc,
                taps=taps,
                tap_count=op.in_features,
                cfg=cfg,
            )
        )

    return _hardware_payload(
        kind="linear",
        input_shape=[op.in_features],
        weight_shape=list(weight.shape),
        bias_shape=list(bias.shape),
        output_shape=[op.out_features],
        weight=weight,
        bias=bias,
        output_count=op.out_features,
        output_entries=output_entries,
        cfg=cfg,
    )


def _conv2d_hardware_payload(
    inputs: np.ndarray,
    op: QuantizedConv2dIR,
    cfg: FixedPointConfig,
) -> dict[str, Any]:
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
    output_entries: list[dict[str, Any]] = []
    flat_index = 0

    for out_channel in range(op.out_channels):
        for out_y in range(op.output_height):
            for out_x in range(op.output_width):
                if len(output_entries) >= HARDWARE_OUTPUT_ENTRY_LIMIT:
                    break
                bias_raw = int(bias[out_channel])
                acc = bias_raw << cfg.frac_bits
                taps: list[dict[str, Any]] = []
                tap_index = 0
                for in_channel in range(op.in_channels):
                    for kernel_y in range(op.kernel_height):
                        in_y = out_y * op.stride[0] + kernel_y - op.padding[0]
                        if in_y < 0 or in_y >= op.input_height:
                            continue
                        for kernel_x in range(op.kernel_width):
                            in_x = out_x * op.stride[1] + kernel_x - op.padding[1]
                            if in_x < 0 or in_x >= op.input_width:
                                continue
                            input_raw = int(data[in_channel, in_y, in_x])
                            weight_raw = int(
                                weight[out_channel, in_channel, kernel_y, kernel_x]
                            )
                            product_raw = input_raw * weight_raw
                            acc += product_raw
                            if len(taps) < HARDWARE_TAPS_PER_ENTRY_LIMIT:
                                taps.append(
                                    {
                                        "tap_index": tap_index,
                                        "input_index": [in_channel, in_y, in_x],
                                        "weight_index": [
                                            out_channel,
                                            in_channel,
                                            kernel_y,
                                            kernel_x,
                                        ],
                                        "input_raw": input_raw,
                                        "weight_raw": weight_raw,
                                        "product_raw": product_raw,
                                        "accumulator_raw": acc,
                                    }
                                )
                            tap_index += 1
                output_entries.append(
                    _output_entry_payload(
                        index=[out_channel, out_y, out_x],
                        flat_index=flat_index,
                        bias_raw=bias_raw,
                        accumulator_raw=acc,
                        taps=taps,
                        tap_count=tap_index,
                        cfg=cfg,
                    )
                )
                flat_index += 1
            if len(output_entries) >= HARDWARE_OUTPUT_ENTRY_LIMIT:
                break
        if len(output_entries) >= HARDWARE_OUTPUT_ENTRY_LIMIT:
            break

    output_count = op.out_channels * op.output_height * op.output_width
    return _hardware_payload(
        kind="conv2d",
        input_shape=list(data.shape),
        weight_shape=list(weight.shape),
        bias_shape=list(bias.shape),
        output_shape=[op.out_channels, op.output_height, op.output_width],
        weight=weight,
        bias=bias,
        output_count=output_count,
        output_entries=output_entries,
        cfg=cfg,
    )


def _hardware_payload(
    *,
    kind: str,
    input_shape: list[int],
    weight_shape: list[int],
    bias_shape: list[int],
    output_shape: list[int],
    weight: np.ndarray,
    bias: np.ndarray,
    output_count: int,
    output_entries: list[dict[str, Any]],
    cfg: FixedPointConfig,
) -> dict[str, Any]:
    weights, weights_truncated = _bounded_flat_values(
        weight,
        HARDWARE_PARAMETER_VALUE_LIMIT,
    )
    biases, biases_truncated = _bounded_flat_values(
        bias,
        HARDWARE_PARAMETER_VALUE_LIMIT,
    )
    return {
        "kind": kind,
        "fixed_point": {
            "bits": cfg.bits,
            "frac_bits": cfg.frac_bits,
            "acc_bits": cfg.acc_bits,
        },
        "input_shape": input_shape,
        "weight_shape": weight_shape,
        "bias_shape": bias_shape,
        "output_shape": output_shape,
        "weights": weights,
        "weight_count": int(np.asarray(weight).size),
        "weights_truncated": weights_truncated,
        "biases": biases,
        "bias_count": int(np.asarray(bias).size),
        "biases_truncated": biases_truncated,
        "output_entry_count": output_count,
        "output_entries": output_entries,
        "output_entries_truncated": len(output_entries) < output_count,
    }


def _output_entry_payload(
    *,
    index: list[int],
    flat_index: int,
    bias_raw: int,
    accumulator_raw: int,
    taps: list[dict[str, Any]],
    tap_count: int,
    cfg: FixedPointConfig,
) -> dict[str, Any]:
    shifted_raw = accumulator_raw >> cfg.frac_bits
    saturated_output_raw = saturate_int(shifted_raw, cfg.bits)
    return {
        "index": index,
        "flat_index": flat_index,
        "bias_raw": bias_raw,
        "initial_accumulator_raw": bias_raw << cfg.frac_bits,
        "tap_count": tap_count,
        "taps": taps,
        "taps_truncated": len(taps) < tap_count,
        "accumulator_raw": accumulator_raw,
        "shifted_raw": shifted_raw,
        "saturated_output_raw": saturated_output_raw,
        "output_raw": saturated_output_raw,
        "saturation_flag": saturated_output_raw != shifted_raw,
    }


def _bounded_flat_values(values: np.ndarray, limit: int) -> tuple[list[int], bool]:
    data = np.asarray(values, dtype=np.int64).reshape(-1)
    return (
        [int(value) for value in data[:limit]],
        data.size > limit,
    )


def _validate_vector_index(inputs: np.ndarray, vector_index: int) -> None:
    if vector_index < 0 or vector_index >= len(inputs):
        raise IndexError(f"vector_index {vector_index} outside 0..{len(inputs) - 1}")


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
