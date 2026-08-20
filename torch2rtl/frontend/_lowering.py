from __future__ import annotations

import math
from collections.abc import Sequence as SequenceABC
from typing import Sequence

from torch2rtl.frontend._errors import UnsupportedOpError
from torch2rtl.ir.graph import GraphIR
from torch2rtl.ir.ops import ArgmaxIR, Conv2dIR, FlattenIR, LinearIR, ReluIR
from torch2rtl.ir.tensor import TensorIR


_MAX_TENSOR_ELEMENTS = 1_000_000
_MAX_TENSOR_RANK = 64
_SV_INT_MAX = (1 << 31) - 1


def _require_exact_string_dict(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise UnsupportedOpError(f"Unsupported modified {label} mapping")
    for key in value:
        if type(key) is not str:
            raise UnsupportedOpError(f"Unsupported modified {label} keys")
    return value


def _raw_static_attribute(owner: object, name: str) -> object:
    namespace = object.__getattribute__(owner, "__dict__")
    if type(namespace) is dict:
        _require_exact_string_dict(namespace, "runtime namespace")
        if name in namespace:
            return namespace[name]
        raise AttributeError(name)
    for cls in type.__getattribute__(owner, "__mro__"):
        namespace = type.__getattribute__(cls, "__dict__")
        if name in namespace:
            return namespace[name]
    raise AttributeError(name)


def lower_fx_graph(
    *,
    model: object,
    traced: object,
    modules: dict[str, object],
    input_shape: tuple[int, ...],
    float_dtype: str,
    transformations: list[dict[str, str]],
    nn: object,
    torch: object,
) -> GraphIR:
    has_batchnorm_fusion = bool(transformations)
    input_tensor = TensorIR(name="input", shape=input_shape, dtype=float_dtype)
    current_tensor = input_tensor
    current_node: object | None = None
    ops: list[object] = []
    saw_argmax = False
    saw_output = False
    source_current_shape = (1, *input_shape) if has_batchnorm_fusion else None

    for node in traced.graph.nodes:
        if node.op == "placeholder":
            if current_node is not None:
                raise UnsupportedOpError(
                    "Unsupported FX graph: exactly one tensor input is required"
                )
            current_node = node
            continue
        if node.op == "output":
            output_node = _parse_output_node(node)
            if output_node is not current_node:
                raise UnsupportedOpError(
                    "Unsupported FX graph: output must be the final node of a "
                    "single sequential chain"
                )
            saw_output = True
            continue
        if saw_output:
            raise UnsupportedOpError("Unsupported FX graph: node found after output")
        if current_node is None:
            raise UnsupportedOpError(
                "Unsupported FX graph: operation appears before the input placeholder"
            )
        if saw_argmax:
            raise UnsupportedOpError(
                "Unsupported FX graph: argmax must be the final operation"
            )
        if node.op == "call_module":
            _require_sequential_input(node, current_node)
            module = modules[str(node.target)]
            if source_current_shape is not None and type(module) is nn.Flatten:
                start_dim = _module_integer(
                    _required_module_attribute(module, "Flatten", "start_dim"),
                    "Flatten",
                    "start_dim",
                )
                end_dim = _module_integer(
                    _required_module_attribute(module, "Flatten", "end_dim"),
                    "Flatten",
                    "end_dim",
                )
                try:
                    source_flatten_shape = _flatten_shape(
                        source_current_shape, start_dim, end_dim
                    )
                    fused_flatten_shape = _flatten_shape(
                        current_tensor.shape, start_dim, end_dim
                    )
                except UnsupportedOpError as exc:
                    raise UnsupportedOpError(
                        "Unsupported Flatten with singleton batch adapter: "
                        "dimensions must be valid for both source NCHW and "
                        "fused CHW shapes"
                    ) from exc
                if source_flatten_shape != (1, *fused_flatten_shape):
                    raise UnsupportedOpError(
                        "Unsupported Flatten with singleton batch adapter: source "
                        f"shape {source_flatten_shape} does not preserve the fused "
                        f"batchless shape {(1, *fused_flatten_shape)}"
                    )
            current_tensor, new_op = _parse_module_node(
                node_name=_safe_name(str(node.name)),
                module=module,
                current_tensor=current_tensor,
                nn=nn,
            )
            if source_current_shape is not None:
                source_current_shape = (1, *current_tensor.shape)
            ops.append(new_op)
            current_node = node
            continue
        if node.op == "call_function" and node.target is torch.argmax:
            _require_sequential_input(node, current_node, allow_parameters=True)
            current_tensor, argmax_op = _lower_argmax_node(
                node, current_tensor, is_method=False
            )
            ops.append(argmax_op)
            saw_argmax = True
            current_node = node
            continue
        if node.op == "call_method" and str(node.target) == "argmax":
            _require_sequential_input(node, current_node, allow_parameters=True)
            current_tensor, argmax_op = _lower_argmax_node(
                node, current_tensor, is_method=True
            )
            ops.append(argmax_op)
            saw_argmax = True
            current_node = node
            continue
        raise UnsupportedOpError(
            f"Unsupported FX node/op/module: op={node.op}, target={node.target}"
        )

    if current_node is None:
        raise UnsupportedOpError("Unsupported FX graph: missing input placeholder")
    if not saw_output:
        raise UnsupportedOpError("Unsupported FX graph: missing output node")

    metadata: dict[str, object] = {"source": type(model).__name__}
    if has_batchnorm_fusion:
        metadata["input_adapter"] = {"kind": "singleton_batch_n1"}
        metadata["transformations"] = transformations
    return GraphIR(
        input=input_tensor,
        output=current_tensor,
        ops=tuple(ops),
        metadata=metadata,
    )


def _parse_module_node(
    node_name: str,
    module: object,
    current_tensor: TensorIR,
    nn: object,
) -> tuple[TensorIR, object]:
    if isinstance(module, nn.Linear):
        return _lower_linear_module(node_name, module, current_tensor, nn)

    if isinstance(module, nn.Conv2d):
        return _lower_conv2d_module(node_name, module, current_tensor, nn)

    if isinstance(module, nn.ReLU):
        return _lower_relu_module(node_name, module, current_tensor)

    if isinstance(module, nn.Flatten):
        return _lower_flatten_module(node_name, module, current_tensor)

    raise UnsupportedOpError(
        f"Unsupported FX node/op/module: module={type(module).__name__}"
    )


def _lower_linear_module(
    node_name: str,
    module: object,
    current_tensor: TensorIR,
    nn: object,
) -> tuple[TensorIR, LinearIR]:
    module_in_features = _positive_module_integer(
        _required_module_attribute(module, "Linear", "in_features"),
        "Linear",
        "in_features",
    )
    module_out_features = _positive_module_integer(
        _required_module_attribute(module, "Linear", "out_features"),
        "Linear",
        "out_features",
    )
    weight = _required_parameter(
        _required_module_attribute(module, "Linear", "weight"),
        "Linear",
        "weight",
        nn,
    )
    bias = _optional_parameter(
        _required_module_attribute(module, "Linear", "bias"),
        "Linear",
        "bias",
        nn,
    )
    if len(current_tensor.shape) != 1:
        raise UnsupportedOpError(
            "Unsupported Linear input shape: only a 1-D feature vector is "
            f"supported, got {current_tensor.shape}; flatten explicitly first"
        )
    in_features = int(current_tensor.shape[0])
    if in_features != module_in_features:
        raise UnsupportedOpError(
            "Unsupported Linear input shape: "
            f"expected ({module.in_features},), got {current_tensor.shape}"
        )
    expected_weight_shape = (module_out_features, module_in_features)
    if tuple(weight.shape) != expected_weight_shape:
        raise UnsupportedOpError(
            "Unsupported Linear weight shape: "
            f"expected {expected_weight_shape}, got {tuple(weight.shape)}"
        )
    if bias is not None and tuple(bias.shape) != (module_out_features,):
        raise UnsupportedOpError(
            "Unsupported Linear bias shape: "
            f"expected ({module.out_features},), got {tuple(module.bias.shape)}"
        )
    output = TensorIR(
        name=f"{node_name}_out",
        shape=(module_out_features,),
        dtype=current_tensor.dtype,
    )
    bias_values = None if bias is None else bias.detach().cpu().numpy().copy()
    return output, LinearIR(
        name=node_name,
        input=current_tensor,
        output=output,
        in_features=module_in_features,
        out_features=module_out_features,
        weight=weight.detach().cpu().numpy().copy(),
        bias=bias_values,
    )


def _lower_conv2d_module(
    node_name: str,
    module: object,
    current_tensor: TensorIR,
    nn: object,
) -> tuple[TensorIR, Conv2dIR]:
    module_in_channels = _positive_module_integer(
        _required_module_attribute(module, "Conv2d", "in_channels"),
        "Conv2d",
        "in_channels",
    )
    module_out_channels = _positive_module_integer(
        _required_module_attribute(module, "Conv2d", "out_channels"),
        "Conv2d",
        "out_channels",
    )
    groups = _positive_module_integer(
        _required_module_attribute(module, "Conv2d", "groups"),
        "Conv2d",
        "groups",
    )
    weight = _required_parameter(
        _required_module_attribute(module, "Conv2d", "weight"),
        "Conv2d",
        "weight",
        nn,
    )
    bias = _optional_parameter(
        _required_module_attribute(module, "Conv2d", "bias"),
        "Conv2d",
        "bias",
        nn,
    )
    if len(current_tensor.shape) == 4:
        raise UnsupportedOpError(
            "Unsupported Conv2d input: batch dimension is not supported; "
            "use unbatched (C, H, W)"
        )
    if len(current_tensor.shape) != 3:
        raise UnsupportedOpError(
            f"Unsupported Conv2d input rank: expected (C, H, W), got {current_tensor.shape}"
        )
    if groups != 1:
        raise UnsupportedOpError("Unsupported Conv2d groups: only groups=1 is supported")
    padding_mode = _required_module_attribute(module, "Conv2d", "padding_mode")
    if type(padding_mode) is not str or padding_mode != "zeros":
        raise UnsupportedOpError(
            "Unsupported Conv2d padding_mode: only zero padding is supported"
        )
    dilation = _int_pair(
        _required_module_attribute(module, "Conv2d", "dilation"),
        "dilation",
    )
    if dilation != (1, 1):
        raise UnsupportedOpError("Unsupported Conv2d dilation: only dilation=1 is supported")
    padding = _int_pair(
        _required_module_attribute(module, "Conv2d", "padding"),
        "padding",
    )
    stride = _int_pair(
        _required_module_attribute(module, "Conv2d", "stride"),
        "stride",
    )
    kernel_height, kernel_width = _int_pair(
        _required_module_attribute(module, "Conv2d", "kernel_size"),
        "kernel_size",
    )
    _validate_conv_pair("kernel_size", (kernel_height, kernel_width), minimum=1)
    _validate_conv_pair("stride", stride, minimum=1)
    _validate_conv_pair("padding", padding, minimum=0)
    for attribute_name, values in (
        ("kernel_size", (kernel_height, kernel_width)),
        ("stride", stride),
        ("padding", padding),
    ):
        if any(value > _SV_INT_MAX for value in values):
            raise UnsupportedOpError(
                f"Unsupported Conv2d {attribute_name}: values must fit a "
                "signed 32-bit SystemVerilog int"
            )
    in_channels, input_height, input_width = (
        int(dim) for dim in current_tensor.shape
    )
    if in_channels != module_in_channels:
        raise UnsupportedOpError(
            "Unsupported Conv2d shape: "
            f"expected {module.in_channels} input channels, got {in_channels}"
        )
    output_height = _conv_output_dim(
        input_height,
        kernel_height,
        stride[0],
        padding[0],
    )
    output_width = _conv_output_dim(
        input_width,
        kernel_width,
        stride[1],
        padding[1],
    )
    if output_height <= 0 or output_width <= 0:
        raise UnsupportedOpError(
            "Unsupported Conv2d output shape: "
            f"got ({module.out_channels}, {output_height}, {output_width})"
        )
    _validate_tensor_element_count(
        (module_out_channels, output_height, output_width),
        "Conv2d output",
    )
    for output_size, step, kernel_size in (
        (output_height, stride[0], kernel_height),
        (output_width, stride[1], kernel_width),
    ):
        maximum_positive_coordinate = (output_size - 1) * step + kernel_size - 1
        if maximum_positive_coordinate > _SV_INT_MAX:
            raise UnsupportedOpError(
                "Unsupported Conv2d coordinates: index arithmetic must fit a "
                "signed 32-bit SystemVerilog int"
            )
    expected_weight_shape = (
        module_out_channels,
        in_channels,
        kernel_height,
        kernel_width,
    )
    actual_weight_shape = tuple(int(dim) for dim in weight.shape)
    if actual_weight_shape != expected_weight_shape:
        raise UnsupportedOpError(
            "Unsupported Conv2d weight shape: "
            f"expected {expected_weight_shape}, got {actual_weight_shape}"
        )
    if bias is not None and tuple(bias.shape) != (module_out_channels,):
        raise UnsupportedOpError(
            "Unsupported Conv2d bias shape: "
            f"expected ({module.out_channels},), got {tuple(module.bias.shape)}"
        )
    output = TensorIR(
        name=f"{node_name}_out",
        shape=(module_out_channels, output_height, output_width),
        dtype=current_tensor.dtype,
    )
    bias_values = None if bias is None else bias.detach().cpu().numpy().copy()
    return output, Conv2dIR(
        name=node_name,
        input=current_tensor,
        output=output,
        in_channels=in_channels,
        out_channels=module_out_channels,
        input_height=input_height,
        input_width=input_width,
        output_height=output_height,
        output_width=output_width,
        kernel_height=kernel_height,
        kernel_width=kernel_width,
        stride=stride,
        padding=padding,
        weight=weight.detach().cpu().numpy().copy(),
        bias=bias_values,
    )


def _lower_relu_module(
    node_name: str,
    module: object,
    current_tensor: TensorIR,
) -> tuple[TensorIR, ReluIR]:
    inplace = _required_module_attribute(module, "ReLU", "inplace")
    if type(inplace) is not bool:
        raise UnsupportedOpError(
            "Unsupported ReLU inplace: expected a built-in boolean"
        )
    if inplace:
        raise UnsupportedOpError(
            "Unsupported ReLU inplace=True: input mutation is not lowered"
        )
    output = TensorIR(
        name=f"{node_name}_out",
        shape=current_tensor.shape,
        dtype=current_tensor.dtype,
    )
    return output, ReluIR(name=node_name, input=current_tensor, output=output)


def _lower_flatten_module(
    node_name: str,
    module: object,
    current_tensor: TensorIR,
) -> tuple[TensorIR, FlattenIR]:
    start_dim = _module_integer(
        _required_module_attribute(module, "Flatten", "start_dim"),
        "Flatten",
        "start_dim",
    )
    end_dim = _module_integer(
        _required_module_attribute(module, "Flatten", "end_dim"),
        "Flatten",
        "end_dim",
    )
    output_shape = _flatten_shape(
        shape=current_tensor.shape,
        start_dim=start_dim,
        end_dim=end_dim,
    )
    if len(output_shape) != 1:
        raise UnsupportedOpError(
            "Unsupported Flatten: only flattening the complete tensor to one "
            f"dimension is supported, got output shape {output_shape}"
        )
    output = TensorIR(
        name=f"{node_name}_out",
        shape=output_shape,
        dtype=current_tensor.dtype,
    )
    return output, FlattenIR(name=node_name, input=current_tensor, output=output)


def _flatten_shape(shape: tuple[int, ...], start_dim: int, end_dim: int) -> tuple[int, ...]:
    rank = len(shape)
    if rank == 0:
        return shape
    if start_dim < 0:
        start_dim += rank
    if end_dim < 0:
        end_dim += rank
    if start_dim < 0 or end_dim >= rank or start_dim > end_dim:
        raise UnsupportedOpError(
            f"Unsupported Flatten dims: start_dim={start_dim}, end_dim={end_dim}"
        )
    flattened = math.prod(shape[start_dim : end_dim + 1])
    return (*shape[:start_dim], flattened, *shape[end_dim + 1 :])



def _validate_input_shape(input_shape: Sequence[int]) -> tuple[int, ...]:
    if type(input_shape) in (str, bytes, bytearray) or not isinstance(
        input_shape,
        SequenceABC,
    ):
        raise UnsupportedOpError(
            "Unsupported input shape: expected a finite sequence of positive integers"
        )
    try:
        shape = tuple(input_shape)
    except Exception as exc:
        raise UnsupportedOpError(
            "Unsupported input shape: expected a finite sequence of positive integers"
        ) from exc
    if len(shape) > _MAX_TENSOR_RANK:
        raise UnsupportedOpError(
            f"Unsupported input shape: rank must not exceed {_MAX_TENSOR_RANK}"
        )
    for index, dim in enumerate(shape):
        if type(dim) is not int or dim <= 0:
            raise UnsupportedOpError(
                "Unsupported input shape: positive integers are required; "
                f"dimension at index {index} is not a positive built-in int"
            )
    _validate_tensor_element_count(shape, "input shape")
    return shape


def _validate_tensor_element_count(shape: Sequence[int], label: str) -> None:
    element_count = 1
    for dimension in shape:
        if dimension > _MAX_TENSOR_ELEMENTS // element_count:
            raise UnsupportedOpError(
                f"Unsupported {label}: total element count exceeds "
                f"the v0.2 limit of {_MAX_TENSOR_ELEMENTS}"
            )
        element_count *= dimension


def _parse_output_node(node: object) -> object:
    args = tuple(getattr(node, "args", ()))
    kwargs = dict(getattr(node, "kwargs", {}))
    if len(args) != 1 or kwargs:
        raise UnsupportedOpError(
            "Unsupported FX output: expected one tensor value without keyword arguments"
        )
    output_value = args[0]
    if not _is_fx_node(output_value):
        raise UnsupportedOpError(
            "Unsupported FX output: tuples, lists, dictionaries, and constants "
            "are not supported"
        )
    return output_value


def _require_sequential_input(
    node: object,
    expected_node: object,
    allow_parameters: bool = False,
) -> None:
    args = tuple(getattr(node, "args", ()))
    kwargs = dict(getattr(node, "kwargs", {}))
    if not args or not _is_fx_node(args[0]):
        raise UnsupportedOpError(
            f"Unsupported FX arguments for node {getattr(node, 'name', '<unknown>')}: "
            "the first argument must be the previous tensor"
        )
    if args[0] is not expected_node:
        raise UnsupportedOpError(
            f"Unsupported FX graph at node {getattr(node, 'name', '<unknown>')}: "
            "only a single sequential dependency chain is supported"
        )
    if not allow_parameters and (len(args) != 1 or kwargs):
        raise UnsupportedOpError(
            f"Unsupported FX arguments for node {getattr(node, 'name', '<unknown>')}: "
            "module calls must receive only the previous tensor"
        )


def _validate_argmax_arguments(node: object, is_method: bool) -> None:
    args = tuple(getattr(node, "args", ()))
    kwargs = dict(getattr(node, "kwargs", {}))
    positional = args[1:]
    if len(positional) > 2:
        raise UnsupportedOpError("Unsupported argmax arguments")

    allowed_kwargs = {"dim", "keepdim"}
    if not is_method:
        allowed_kwargs.add("out")
    unexpected = set(kwargs) - allowed_kwargs
    if unexpected:
        names = ", ".join(sorted(str(name) for name in unexpected))
        raise UnsupportedOpError(f"Unsupported argmax keyword arguments: {names}")

    dim = _positional_or_keyword(positional, 0, kwargs, "dim", None)
    keepdim = _positional_or_keyword(positional, 1, kwargs, "keepdim", False)
    if not is_method and kwargs.get("out") is not None:
        raise UnsupportedOpError("Unsupported argmax out argument")
    if dim is not None or keepdim is not False:
        raise UnsupportedOpError(
            "Unsupported argmax: only global argmax with dim=None and "
            "keepdim=False is supported"
        )


def _lower_argmax_node(
    node: object,
    current_tensor: TensorIR,
    *,
    is_method: bool,
) -> tuple[TensorIR, ArgmaxIR]:
    _validate_argmax_arguments(node, is_method=is_method)
    node_name = _safe_name(str(node.name))
    output = TensorIR(name=node_name, shape=(), dtype="int64")
    return output, ArgmaxIR(
        name=node_name,
        input=current_tensor,
        output=output,
    )


def _positional_or_keyword(
    positional: tuple[object, ...],
    index: int,
    kwargs: dict[object, object],
    name: str,
    default: object,
) -> object:
    if index < len(positional):
        if name in kwargs:
            raise UnsupportedOpError(f"argmax received {name} more than once")
        return positional[index]
    return kwargs.get(name, default)


def _is_fx_node(value: object) -> bool:
    return (
        hasattr(value, "op")
        and hasattr(value, "target")
        and hasattr(value, "args")
        and hasattr(value, "kwargs")
    )


def _conv_output_dim(input_size: int, kernel_size: int, stride: int, padding: int) -> int:
    return ((input_size + 2 * padding - kernel_size) // stride) + 1


def _int_pair(value: object, name: str) -> tuple[int, int]:
    if isinstance(value, str):
        raise UnsupportedOpError(f"Unsupported Conv2d {name}: string values are not supported")
    if type(value) is int:
        return (value, value)
    if type(value) is tuple and len(value) == 2:
        first, second = value
        if type(first) is int and type(second) is int:
            return (first, second)
    raise UnsupportedOpError(f"Unsupported Conv2d {name}: expected int or int pair")


def _positive_module_integer(value: object, module: str, attribute: str) -> int:
    normalized = _module_integer(value, module, attribute)
    if normalized <= 0:
        raise UnsupportedOpError(
            f"Unsupported {module} {attribute}: expected a positive built-in integer, "
            f"got {value!r}"
        )
    return normalized


def _module_integer(value: object, module: str, attribute: str) -> int:
    if type(value) is not int:
        raise UnsupportedOpError(
            f"Unsupported {module} {attribute}: expected a built-in integer, "
            f"got {value!r}"
        )
    return value


def _required_parameter(
    value: object,
    module: str,
    attribute: str,
    nn: object,
) -> object:
    if type(value) is not nn.Parameter:
        raise UnsupportedOpError(
            f"Unsupported {module} {attribute}: expected a Parameter"
        )
    return value


def _required_module_attribute(
    module_value: object,
    module: str,
    attribute: str,
) -> object:
    try:
        return getattr(module_value, attribute)
    except AttributeError as exc:
        raise UnsupportedOpError(
            f"Unsupported {module}: required attribute {attribute!r} is missing"
        ) from exc


def _optional_parameter(
    value: object,
    module: str,
    attribute: str,
    nn: object,
) -> object | None:
    if value is None:
        return None
    return _required_parameter(value, module, attribute, nn)


def _validate_conv_pair(
    name: str,
    values: tuple[int, int],
    minimum: int,
) -> None:
    if any(value < minimum for value in values):
        relation = "positive" if minimum == 1 else "non-negative"
        raise UnsupportedOpError(
            f"Unsupported Conv2d {name}: values must be {relation}, got {values}"
        )


def _safe_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name)
    if cleaned and cleaned[0].isdigit():
        cleaned = f"op_{cleaned}"
    return cleaned or "op"



__all__ = ["lower_fx_graph"]
