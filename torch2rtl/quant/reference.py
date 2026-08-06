from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import torch as _torch

from torch2rtl.ir.graph import GraphIR
from torch2rtl.ir.ops import ArgmaxIR, Conv2dIR, FlattenIR, LinearIR, ReluIR
from torch2rtl.quant.fixed_point import FixedPointConfig, quantize_array, saturate_int


_FUNCTIONAL_LINEAR = _torch.nn.functional.linear
_FUNCTIONAL_CONV2D = _torch.nn.functional.conv2d
_FUNCTIONAL_RELU = _torch.nn.functional.relu
del _torch


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
    semantic: bool = True


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
    output_is_class_id: bool = False

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
    output: np.ndarray
    activations: tuple["ActivationResult", ...] = ()


@dataclass(frozen=True)
class FloatReferenceResult:
    logits: np.ndarray
    class_id: int
    output: np.ndarray
    activations: tuple["ActivationResult", ...] = ()


@dataclass(frozen=True)
class ActivationResult:
    name: str
    values: np.ndarray


def _check_accumulator_range(
    value: int,
    cfg: FixedPointConfig,
) -> int:
    if value < cfg.acc_min_int or value > cfg.acc_max_int:
        raise OverflowError(
            f"Value {value} does not fit signed {cfg.acc_bits}-bit "
            f"accumulator range "
            f"[{cfg.acc_min_int}, {cfg.acc_max_int}]"
        )
    return value


def quantize_graph(graph: GraphIR, cfg: FixedPointConfig) -> QuantizedGraph:
    qops: list[QuantizedOp] = []
    saw_argmax = False
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
            saw_argmax = True
        else:
            raise TypeError(f"Unsupported IR op for quantization: {type(op).__name__}")
    if not saw_argmax:
        # PyTorch models commonly return logits.  The RTL exposes those logits
        # unchanged and adds class_id as a derived convenience output.  Keep
        # that hardware-only argmax out of semantic GraphIR.
        qops.append(QuantizedArgmaxIR(name="class_id", semantic=False))
    qgraph = QuantizedGraph(
        input_shape=graph.input.shape,
        ops=tuple(qops),
        cfg=cfg,
        output_is_class_id=saw_argmax,
    )
    validate_accumulator_width(qgraph)
    return qgraph


def validate_accumulator_width(qgraph: QuantizedGraph) -> None:
    """Reject a graph whose exact RTL MAC can wrap for any input value."""
    for op in qgraph.ops:
        if isinstance(op, QuantizedLinearIR):
            required_bits, location, bounds = _linear_accumulator_requirement(
                op,
                qgraph.cfg,
            )
        elif isinstance(op, QuantizedConv2dIR):
            required_bits, location, bounds = _conv2d_accumulator_requirement(
                op,
                qgraph.cfg,
            )
        else:
            continue
        if required_bits > qgraph.cfg.acc_bits:
            lower, upper = bounds
            raise OverflowError(
                f"ACC_BITS={qgraph.cfg.acc_bits} is unsafe for {type(op).__name__} "
                f"'{op.name}' at {location}: possible range [{lower}, {upper}] "
                f"requires at least {required_bits} signed bits"
            )


def _linear_accumulator_requirement(
    op: QuantizedLinearIR,
    cfg: FixedPointConfig,
) -> tuple[int, str, tuple[int, int]]:
    weights = _normalize_fixed_data("Linear weight", op.weight, cfg).reshape(
        op.out_features,
        op.in_features,
    )
    biases = _normalize_fixed_data("Linear bias", op.bias, cfg).reshape(op.out_features)
    worst = (1, "bias", (0, 0))
    for out_idx in range(op.out_features):
        acc_lower = int(biases[out_idx]) << cfg.frac_bits
        acc_upper = acc_lower
        worst = _wider_requirement(
            worst,
            f"output {out_idx} bias shift",
            (acc_lower, acc_upper),
        )
        for in_idx in range(op.in_features):
            product_bounds = _product_bounds(int(weights[out_idx, in_idx]), cfg)
            worst = _wider_requirement(
                worst,
                f"output {out_idx} product {in_idx}",
                product_bounds,
            )
            acc_lower += product_bounds[0]
            acc_upper += product_bounds[1]
            worst = _wider_requirement(
                worst,
                f"output {out_idx} MAC prefix {in_idx + 1}",
                (acc_lower, acc_upper),
            )
    return worst


def _conv2d_accumulator_requirement(
    op: QuantizedConv2dIR,
    cfg: FixedPointConfig,
) -> tuple[int, str, tuple[int, int]]:
    weights = _normalize_fixed_data("Conv2d weight", op.weight, cfg).reshape(
        op.out_channels,
        op.in_channels,
        op.kernel_height,
        op.kernel_width,
    )
    biases = _normalize_fixed_data("Conv2d bias", op.bias, cfg).reshape(op.out_channels)
    worst = (1, "bias", (0, 0))
    for out_channel in range(op.out_channels):
        for out_y in range(op.output_height):
            for out_x in range(op.output_width):
                acc_lower = int(biases[out_channel]) << cfg.frac_bits
                acc_upper = acc_lower
                output_name = f"output ({out_channel}, {out_y}, {out_x})"
                worst = _wider_requirement(
                    worst,
                    f"{output_name} bias shift",
                    (acc_lower, acc_upper),
                )
                prefix = 0
                for in_channel in range(op.in_channels):
                    for kernel_y in range(op.kernel_height):
                        in_y = out_y * op.stride[0] + kernel_y - op.padding[0]
                        if in_y < 0 or in_y >= op.input_height:
                            continue
                        for kernel_x in range(op.kernel_width):
                            in_x = out_x * op.stride[1] + kernel_x - op.padding[1]
                            if in_x < 0 or in_x >= op.input_width:
                                continue
                            product_bounds = _product_bounds(
                                int(
                                    weights[
                                        out_channel,
                                        in_channel,
                                        kernel_y,
                                        kernel_x,
                                    ]
                                ),
                                cfg,
                            )
                            prefix += 1
                            worst = _wider_requirement(
                                worst,
                                f"{output_name} product {prefix}",
                                product_bounds,
                            )
                            acc_lower += product_bounds[0]
                            acc_upper += product_bounds[1]
                            worst = _wider_requirement(
                                worst,
                                f"{output_name} MAC prefix {prefix}",
                                (acc_lower, acc_upper),
                            )
    return worst


def _product_bounds(weight: int, cfg: FixedPointConfig) -> tuple[int, int]:
    products = (cfg.min_int * weight, cfg.max_int * weight)
    return (min(products), max(products))


def _wider_requirement(
    current: tuple[int, str, tuple[int, int]],
    location: str,
    bounds: tuple[int, int],
) -> tuple[int, str, tuple[int, int]]:
    required_bits = _required_signed_bits(*bounds)
    if required_bits > current[0]:
        return (required_bits, location, bounds)
    return current


def _required_signed_bits(lower: int, upper: int) -> int:
    bits = 1
    while lower < -(1 << (bits - 1)) or upper > (1 << (bits - 1)) - 1:
        bits += 1
    return bits


def linear_fixed(
    inputs: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    cfg: FixedPointConfig,
) -> np.ndarray:
    x = _normalize_fixed_data("Linear input", inputs, cfg).reshape(-1)
    w = _normalize_fixed_data("Linear weight", weight, cfg)
    b = _normalize_fixed_data("Linear bias", bias, cfg).reshape(-1)
    if w.ndim != 2:
        raise ValueError(f"Linear weight must be 2-D, got shape {w.shape}")
    if w.shape[1] != x.shape[0]:
        raise ValueError(f"Linear input mismatch: weight {w.shape}, input {x.shape}")
    if b.size != w.shape[0]:
        raise ValueError(f"Linear bias mismatch: weight {w.shape}, bias {b.shape}")
    outputs: list[int] = []
    for out_idx in range(w.shape[0]):
        acc = int(b[out_idx]) << cfg.frac_bits
        acc = _check_accumulator_range(acc, cfg)
        for in_idx in range(w.shape[1]):
            product = int(x[in_idx]) * int(w[out_idx, in_idx])
            product = _check_accumulator_range(product, cfg)
            acc = _check_accumulator_range(acc + product, cfg)
        shifted = acc >> cfg.frac_bits
        outputs.append(saturate_int(shifted, cfg.bits))
    return np.asarray(outputs, dtype=np.int64)


def conv2d_fixed(
    inputs: np.ndarray,
    op: QuantizedConv2dIR,
    cfg: FixedPointConfig,
) -> np.ndarray:
    data = _normalize_fixed_data("Conv2d input", inputs, cfg).reshape(
        op.in_channels,
        op.input_height,
        op.input_width,
    )
    weight = _normalize_fixed_data("Conv2d weight", op.weight, cfg).reshape(
        op.out_channels,
        op.in_channels,
        op.kernel_height,
        op.kernel_width,
    )
    bias = _normalize_fixed_data("Conv2d bias", op.bias, cfg).reshape(op.out_channels)
    output = np.zeros(
        (op.out_channels, op.output_height, op.output_width),
        dtype=np.int64,
    )
    for out_channel in range(op.out_channels):
        for out_y in range(op.output_height):
            for out_x in range(op.output_width):
                acc = int(bias[out_channel]) << cfg.frac_bits
                acc = _check_accumulator_range(acc, cfg)
                for in_channel in range(op.in_channels):
                    for kernel_y in range(op.kernel_height):
                        in_y = out_y * op.stride[0] + kernel_y - op.padding[0]
                        if in_y < 0 or in_y >= op.input_height:
                            continue
                        for kernel_x in range(op.kernel_width):
                            in_x = out_x * op.stride[1] + kernel_x - op.padding[1]
                            if in_x < 0 or in_x >= op.input_width:
                                continue
                            product = int(data[in_channel, in_y, in_x]) * int(
                                weight[out_channel, in_channel, kernel_y, kernel_x]
                            )
                            product = _check_accumulator_range(product, cfg)
                            acc = _check_accumulator_range(acc + product, cfg)
                output[out_channel, out_y, out_x] = saturate_int(
                    acc >> cfg.frac_bits,
                    cfg.bits,
                )
    return output


def infer_float_graph(graph: GraphIR, input_values: np.ndarray) -> FloatReferenceResult:
    import torch

    float_dtype = _graph_float_dtype(graph)
    source = np.asarray(input_values)
    if source.dtype.kind not in "fiu" or (
        source.dtype.kind == "f"
        and source.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
    ):
        raise TypeError(
            "Float GraphIR input must contain float32/float64 or integer values, "
            f"got {source.dtype}"
        )
    logits = source.astype(float_dtype, copy=False)
    if logits.shape != graph.input.shape:
        raise ValueError(
            f"GraphIR input shape mismatch: expected {graph.input.shape}, got {logits.shape}"
        )
    class_id = 0
    saw_argmax = False
    model_output = np.asarray(logits, dtype=float_dtype)
    activations: list[ActivationResult] = []
    for op in graph.ops:
        if logits.shape != op.input.shape:
            raise ValueError(
                f"GraphIR shape mismatch before {op.name}: "
                f"expected {op.input.shape}, got {logits.shape}"
            )
        if isinstance(op, LinearIR):
            data_tensor = torch.from_numpy(
                np.array(logits.reshape(-1), dtype=float_dtype, copy=True)
            )
            weight_tensor = torch.from_numpy(
                np.array(op.weight, dtype=float_dtype, copy=True)
            )
            bias_tensor = (
                torch.from_numpy(np.array(op.bias, dtype=float_dtype, copy=True))
                if op.bias is not None
                else None
            )
            with torch.no_grad():
                output_tensor = _FUNCTIONAL_LINEAR(
                    data_tensor,
                    weight_tensor,
                    bias_tensor,
                )
            logits = output_tensor.detach().cpu().numpy().copy()
        elif isinstance(op, Conv2dIR):
            logits = _conv2d_float(logits, op, float_dtype)
        elif isinstance(op, ReluIR):
            data_tensor = torch.from_numpy(
                np.array(logits, dtype=float_dtype, copy=True)
            )
            with torch.no_grad():
                output_tensor = _FUNCTIONAL_RELU(data_tensor, inplace=False)
            logits = output_tensor.detach().cpu().numpy().copy()
        elif isinstance(op, FlattenIR):
            logits = logits.reshape(-1)
        elif isinstance(op, ArgmaxIR):
            class_id = int(np.argmax(logits))
            saw_argmax = True
            model_output = np.asarray(class_id, dtype=np.int64)
        else:
            raise TypeError(f"Unsupported IR op for float reference: {type(op).__name__}")
        if isinstance(op, ArgmaxIR):
            activation = np.asarray(class_id, dtype=np.int64)
        else:
            if logits.shape != op.output.shape:
                raise ValueError(
                    f"GraphIR shape mismatch after {op.name}: "
                    f"expected {op.output.shape}, got {logits.shape}"
                )
            activation = np.asarray(logits, dtype=float_dtype)
        activations.append(ActivationResult(name=op.name, values=activation.copy()))
    if not saw_argmax:
        class_id = int(np.argmax(logits))
        model_output = np.asarray(logits, dtype=float_dtype)
    return FloatReferenceResult(
        logits=np.asarray(logits, dtype=float_dtype),
        class_id=class_id,
        output=model_output.copy(),
        activations=tuple(activations),
    )


def _conv2d_float(
    inputs: np.ndarray,
    op: Conv2dIR,
    dtype: np.dtype[np.floating],
) -> np.ndarray:
    import torch

    data_tensor = torch.from_numpy(
        np.array(
            inputs,
            dtype=dtype,
            copy=True,
        ).reshape(op.in_channels, op.input_height, op.input_width)
    )
    weight_tensor = torch.from_numpy(
        np.array(op.weight, dtype=dtype, copy=True)
    )
    bias_tensor = (
        torch.from_numpy(np.array(op.bias, dtype=dtype, copy=True))
        if op.bias is not None
        else None
    )
    with torch.no_grad():
        output_tensor = _FUNCTIONAL_CONV2D(
            data_tensor,
            weight_tensor,
            bias_tensor,
            stride=op.stride,
            padding=op.padding,
        )
    return output_tensor.detach().cpu().numpy().copy()


def infer_quantized(qgraph: QuantizedGraph, input_values: np.ndarray) -> ReferenceResult:
    data = _normalize_fixed_data(
        "QuantizedGraph input",
        input_values,
        qgraph.cfg,
    )
    if data.size != qgraph.input_size:
        raise ValueError(
            f"QuantizedGraph input size mismatch: expected {qgraph.input_size}, "
            f"got {data.size}"
        )
    data = data.reshape(qgraph.input_shape)
    logits = data
    class_id = 0
    saw_argmax = False
    activations: list[ActivationResult] = []
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
            saw_argmax = True
        else:
            raise TypeError(f"Unsupported quantized op: {type(op).__name__}")
        if isinstance(op, QuantizedArgmaxIR):
            activation = np.asarray(class_id, dtype=np.int64)
        else:
            activation = np.asarray(logits, dtype=np.int64)
        if not isinstance(op, QuantizedArgmaxIR) or op.semantic:
            activations.append(ActivationResult(name=op.name, values=activation.copy()))
    if not saw_argmax:
        class_id = int(np.argmax(logits))
    output = (
        np.asarray(class_id, dtype=np.int64)
        if qgraph.output_is_class_id
        else np.asarray(logits, dtype=np.int64)
    )
    return ReferenceResult(
        logits=np.asarray(logits, dtype=np.int64),
        class_id=class_id,
        output=output.copy(),
        activations=tuple(activations),
    )


def _normalize_fixed_data(
    name: str,
    values: object,
    cfg: FixedPointConfig,
) -> np.ndarray:
    data = np.asarray(values)
    if data.size == 0:
        raise ValueError(f"{name} must not be empty")
    if data.dtype.kind == "O":
        raise TypeError(f"{name} object dtype is not supported")
    if data.dtype.kind == "b":
        raise TypeError(f"{name} boolean values are not supported")
    if data.dtype.kind == "f":
        if not np.all(np.isfinite(data)):
            raise ValueError(f"{name} values must be finite")
        if not np.all(data == np.trunc(data)):
            raise ValueError(f"{name} values must be integer-valued")
    elif data.dtype.kind not in "iu":
        raise TypeError(
            f"{name} values must use an integer dtype, got {data.dtype}"
        )
    actual_min = int(data.min().item())
    actual_max = int(data.max().item())
    if actual_min < cfg.min_int or actual_max > cfg.max_int:
        raise OverflowError(
            f"{name} values [{actual_min}, {actual_max}] are outside the range for signed "
            f"{cfg.bits}-bit range [{cfg.min_int}, {cfg.max_int}]"
        )
    return data.astype(np.int64)


def _graph_float_dtype(graph: GraphIR) -> np.dtype[np.floating]:
    try:
        return {
            "float32": np.dtype(np.float32),
            "float64": np.dtype(np.float64),
        }[graph.input.dtype]
    except KeyError as exc:
        raise TypeError(f"Unsupported GraphIR floating dtype: {graph.input.dtype}") from exc
