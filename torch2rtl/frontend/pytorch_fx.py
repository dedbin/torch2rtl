from __future__ import annotations

import builtins
import copy
import dis
import importlib.util
import inspect
import math
import sys
from collections import OrderedDict
from collections.abc import Sequence as SequenceABC
from pathlib import Path
from types import CodeType, FunctionType, ModuleType
from typing import Sequence

import numpy as np
import torch as _torch

from torch2rtl.ir.graph import GraphIR
from torch2rtl.ir.ops import ArgmaxIR, Conv2dIR, FlattenIR, LinearIR, ReluIR
from torch2rtl.ir.tensor import TensorIR


_MAX_TENSOR_ELEMENTS = 1_000_000
_MAX_TENSOR_RANK = 64
_SV_INT_MAX = (1 << 31) - 1


def _definition_fingerprint(value: object, seen: set[int] | None = None) -> object:
    active = set() if seen is None else seen
    if value is None or type(value) in (bool, int, float, complex, str, bytes):
        return (type(value), value)
    identity = id(value)
    if identity in active:
        return ("cycle", identity)
    active.add(identity)
    try:
        if isinstance(value, FunctionType):
            closure = tuple(
                _definition_fingerprint(cell.cell_contents, active)
                for cell in (value.__closure__ or ())
            )
            referenced_globals = tuple(
                (name, _definition_fingerprint(value.__globals__[name], active))
                for name in value.__code__.co_names
                if name in value.__globals__
            )
            return (
                "function",
                identity,
                id(value.__code__),
                _definition_fingerprint(value.__defaults__, active),
                _definition_fingerprint(value.__kwdefaults__, active),
                _definition_fingerprint(value.__annotations__, active),
                _definition_fingerprint(value.__dict__, active),
                closure,
                referenced_globals,
            )
        if isinstance(value, (staticmethod, classmethod)):
            return (type(value), _definition_fingerprint(value.__func__, active))
        if isinstance(value, property):
            return (
                "property",
                _definition_fingerprint(value.fget, active),
                _definition_fingerprint(value.fset, active),
                _definition_fingerprint(value.fdel, active),
            )
        if type(value) is dict:
            return (
                "dict",
                tuple(
                    (
                        _definition_fingerprint(key, active),
                        _definition_fingerprint(item, active),
                    )
                    for key, item in value.items()
                ),
            )
        if type(value) in (tuple, list):
            return (
                type(value),
                tuple(_definition_fingerprint(item, active) for item in value),
            )
        if type(value) in (set, frozenset):
            return (
                type(value),
                frozenset(_definition_fingerprint(item, active) for item in value),
            )
        return ("identity", identity)
    finally:
        active.remove(identity)


_TRUSTED_FRAMEWORK_TYPES = (
    _torch.nn.Module,
    _torch.nn.Sequential,
    _torch.nn.Identity,
    _torch.nn.Linear,
    _torch.nn.Conv2d,
    _torch.nn.ReLU,
    _torch.nn.Flatten,
)
_TRUSTED_PARAMETER_TYPE = _torch.nn.Parameter
_TRUSTED_BUILTINS = builtins.__dict__
_TRUSTED_LEN = builtins.len
_TRUSTED_DEEPCOPY = copy.deepcopy
_TRUSTED_DEEPCOPY_CODE = copy.deepcopy.__code__
_TRUSTED_FRAMEWORK_CLASSES = frozenset(
    cls for module_type in _TRUSTED_FRAMEWORK_TYPES for cls in module_type.__mro__
)
_TRUSTED_FX_CLASSES = (
    _torch.fx.Tracer,
    _torch.fx.Graph,
    _torch.fx.Node,
    _torch.fx.GraphModule,
)
_TRUSTED_FRAMEWORK_DEFINITIONS = tuple(
    (
        cls,
        {
            name: (value, _definition_fingerprint(value))
            for name, value in vars(cls).items()
        },
    )
    for cls in (
        *_TRUSTED_FRAMEWORK_CLASSES,
        *_TRUSTED_FX_CLASSES,
    )
)
_TRUSTED_FUNCTION_BINDINGS = (
    (_torch, "nn", _torch.nn),
    (_torch, "fx", _torch.fx),
    (_torch.nn, "functional", _torch.nn.functional),
    (_torch.nn, "modules", _torch.nn.modules),
    (_torch.nn.modules, "module", _torch.nn.modules.module),
    (_torch.nn.functional, "linear", _torch.nn.functional.linear),
    (_torch.nn.functional, "conv2d", _torch.nn.functional.conv2d),
    (_torch.nn.functional, "relu", _torch.nn.functional.relu),
    (_torch, "relu", _torch.relu),
    (_torch, "relu_", _torch.relu_),
    (_torch, "argmax", _torch.argmax),
    (_torch, "from_numpy", _torch.from_numpy),
    (_torch, "no_grad", _torch.no_grad),
    (_torch, "get_default_dtype", _torch.get_default_dtype),
    (_torch, "Tensor", _torch.Tensor),
    (_torch, "device", _torch.device),
    (_torch, "float32", _torch.float32),
    (_torch, "float64", _torch.float64),
    (_torch, "int64", _torch.int64),
    (_torch, "strided", _torch.strided),
    (_torch.fx, "symbolic_trace", _torch.fx.symbolic_trace),
    (_torch.Tensor, "argmax", _torch.Tensor.argmax),
    (_torch.Tensor, "flatten", _torch.Tensor.flatten),
    (_torch.Tensor, "detach", _torch.Tensor.detach),
    (_torch.Tensor, "numpy", _torch.Tensor.numpy),
    (_torch.Tensor, "cpu", _torch.Tensor.cpu),
    (_torch.Tensor, "contiguous", _torch.Tensor.contiguous),
    (_torch.Tensor, "resolve_conj", _torch.Tensor.resolve_conj),
    (_torch.Tensor, "resolve_neg", _torch.Tensor.resolve_neg),
    (_torch.Tensor, "is_contiguous", _torch.Tensor.is_contiguous),
    (_torch.Tensor, "is_conj", _torch.Tensor.is_conj),
    (_torch.Tensor, "is_neg", _torch.Tensor.is_neg),
    (_torch.Tensor, "stride", _torch.Tensor.stride),
    (_torch.Tensor, "storage_offset", _torch.Tensor.storage_offset),
    (_torch.Tensor, "to", _torch.Tensor.to),
    (_torch.Tensor, "device", _torch.Tensor.device),
    (_torch.Tensor, "dtype", _torch.Tensor.dtype),
    (_torch.Tensor, "layout", _torch.Tensor.layout),
    (_torch.Tensor, "shape", _torch.Tensor.shape),
    (_torch.Tensor, "grad", _torch.Tensor.grad),
    (_torch.Tensor, "requires_grad", _torch.Tensor.requires_grad),
)
_TRUSTED_FUNCTION_FINGERPRINTS = tuple(
    (owner, name, function, _definition_fingerprint(function))
    for owner, name, function in _TRUSTED_FUNCTION_BINDINGS
)
del _torch


class UnsupportedOpError(RuntimeError):
    """Raised when the FX graph contains an operation outside the MVP subset."""


def load_model_from_file(path: Path) -> object:
    module = _load_python_module(path)
    if hasattr(module, "create_model"):
        model = module.create_model()
    elif hasattr(module, "model"):
        model = module.model
    else:
        raise ValueError(
            f"{path} must define create_model() or a global variable named model"
        )
    if hasattr(model, "eval"):
        model.eval()
    return model


def parse_model(
    model: object,
    input_shape: Sequence[int],
    input_dtype: object | None = None,
) -> GraphIR:
    import torch
    import torch.nn as nn

    normalized_input_shape = _validate_input_shape(input_shape)
    float_dtype = _validate_model_semantics(model, nn, torch, input_dtype)
    trace_model = _copy_model_for_tracing(model, nn, torch)
    instance_state = _snapshot_module_state(trace_model, nn, torch)
    class_state = _snapshot_module_class_definitions(trace_model, nn)
    trace_error: Exception | None = None
    traced: object | None = None
    try:
        traced = torch.fx.symbolic_trace(trace_model)
    except Exception as exc:
        trace_error = exc
    class_state_changed = _restore_module_class_definitions(class_state)
    instance_state_changed = instance_state != _snapshot_module_state(
        trace_model,
        nn,
        torch,
    )
    if class_state_changed or instance_state_changed:
        raise UnsupportedOpError(
            "Unsupported Python state mutation during FX symbolic trace"
        )
    if trace_error is not None:
        raise UnsupportedOpError(
            f"Unsupported FX symbolic trace semantics: {trace_error}"
        ) from trace_error
    assert traced is not None
    modules = dict(_trusted_named_modules(traced, nn))
    input_tensor = TensorIR(
        name="input",
        shape=normalized_input_shape,
        dtype=float_dtype,
    )
    current_tensor = input_tensor
    current_node: object | None = None
    ops: list[object] = []
    saw_argmax = False
    saw_output = False

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
            current_tensor, new_op = _parse_module_node(
                node_name=_safe_name(str(node.name)),
                module=module,
                current_tensor=current_tensor,
                nn=nn,
            )
            ops.append(new_op)
            current_node = node
            continue
        if node.op == "call_function" and node.target is torch.argmax:
            _require_sequential_input(node, current_node, allow_parameters=True)
            _validate_argmax_arguments(node, is_method=False)
            argmax_input = current_tensor
            current_tensor = TensorIR(name=_safe_name(str(node.name)), shape=(), dtype="int64")
            ops.append(
                ArgmaxIR(
                    name=_safe_name(str(node.name)),
                    input=argmax_input,
                    output=current_tensor,
                )
            )
            saw_argmax = True
            current_node = node
            continue
        if node.op == "call_method" and str(node.target) == "argmax":
            _require_sequential_input(node, current_node, allow_parameters=True)
            _validate_argmax_arguments(node, is_method=True)
            argmax_input = current_tensor
            current_tensor = TensorIR(name=_safe_name(str(node.name)), shape=(), dtype="int64")
            ops.append(
                ArgmaxIR(
                    name=_safe_name(str(node.name)),
                    input=argmax_input,
                    output=current_tensor,
                )
            )
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

    graph = GraphIR(
        input=input_tensor,
        output=current_tensor,
        ops=tuple(ops),
        metadata={"source": type(model).__name__},
    )
    _validate_lowered_semantics(
        model=model,
        graph=graph,
        input_shape=normalized_input_shape,
        float_dtype=float_dtype,
        nn=nn,
        torch=torch,
    )
    return graph


def _parse_module_node(
    node_name: str,
    module: object,
    current_tensor: TensorIR,
    nn: object,
) -> tuple[TensorIR, object]:
    if isinstance(module, nn.Linear):
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
        if bias is not None and tuple(bias.shape) != (
            module_out_features,
        ):
            raise UnsupportedOpError(
                "Unsupported Linear bias shape: "
                f"expected ({module.out_features},), got {tuple(module.bias.shape)}"
            )
        output_shape = (module_out_features,)
        output = TensorIR(
            name=f"{node_name}_out",
            shape=output_shape,
            dtype=current_tensor.dtype,
        )
        bias_values = None
        if bias is not None:
            bias_values = bias.detach().cpu().numpy().copy()
        op = LinearIR(
            name=node_name,
            input=current_tensor,
            output=output,
            in_features=module_in_features,
            out_features=module_out_features,
            weight=weight.detach().cpu().numpy().copy(),
            bias=bias_values,
        )
        return output, op

    if isinstance(module, nn.Conv2d):
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
        in_channels, input_height, input_width = (int(dim) for dim in current_tensor.shape)
        if in_channels != module_in_channels:
            raise UnsupportedOpError(
                "Unsupported Conv2d shape: "
                f"expected {module.in_channels} input channels, got {in_channels}"
            )
        output_height = _conv_output_dim(input_height, kernel_height, stride[0], padding[0])
        output_width = _conv_output_dim(input_width, kernel_width, stride[1], padding[1])
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
        if bias is not None and tuple(bias.shape) != (
            module_out_channels,
        ):
            raise UnsupportedOpError(
                "Unsupported Conv2d bias shape: "
                f"expected ({module.out_channels},), got {tuple(module.bias.shape)}"
            )
        output = TensorIR(
            name=f"{node_name}_out",
            shape=(module_out_channels, output_height, output_width),
            dtype=current_tensor.dtype,
        )
        bias_values = None
        if bias is not None:
            bias_values = bias.detach().cpu().numpy().copy()
        op = Conv2dIR(
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
        return output, op

    if isinstance(module, nn.ReLU):
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

    if isinstance(module, nn.Flatten):
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

    raise UnsupportedOpError(
        f"Unsupported FX node/op/module: module={type(module).__name__}"
    )


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


def _validate_model_semantics(
    model: object,
    nn: object,
    torch: object,
    input_dtype: object | None,
) -> str:
    _validate_framework_integrity(nn)
    if not isinstance(model, nn.Module):
        raise UnsupportedOpError(
            f"Unsupported model type: expected torch.nn.Module, got {type(model).__name__}"
        )

    module_api = torch.nn.modules.module
    for registry_name in (
        "_global_buffer_registration_hooks",
        "_global_module_registration_hooks",
        "_global_parameter_registration_hooks",
        "_global_backward_pre_hooks",
        "_global_backward_hooks",
        "_global_forward_pre_hooks",
        "_global_forward_hooks",
        "_global_forward_hooks_always_called",
        "_global_forward_hooks_with_kwargs",
    ):
        registry = getattr(module_api, registry_name, {})
        if registry:
            raise UnsupportedOpError(
                f"Unsupported global hook registry {registry_name}: "
                "hook semantics are not lowered"
            )

    lowered_types = (nn.Linear, nn.Conv2d, nn.ReLU, nn.Flatten)
    modules = tuple(_trusted_named_modules(model, nn))
    for module_name, module in modules:
        display_name = module_name or "<root>"
        module_type = type(module)
        if type(module_type) is not type:
            type_name = type.__getattribute__(module_type, "__name__")
            raise UnsupportedOpError(
                f"Unsupported custom metaclass on module type {type_name}"
            )
        state = _raw_module_state(module)
        _validate_module_dispatch(module, display_name, nn)
        _validate_no_custom_copy_protocol(type(module), nn)
        _validate_no_custom_class_state(type(module), nn)
        _validate_copy_safe_instance_state(module, display_name, torch)
        _validate_forward_python_state(type(module), nn, torch)
        if state.get("_forward_hooks"):
            raise UnsupportedOpError(
                f"Unsupported forward hook on module {display_name}: "
                "hook semantics are not lowered"
            )
        if state.get("_forward_pre_hooks"):
            raise UnsupportedOpError(
                f"Unsupported forward pre-hook on module {display_name}: "
                "hook semantics are not lowered"
            )
        if isinstance(module, lowered_types):
            if type(module) not in lowered_types:
                raise UnsupportedOpError(
                    f"Unsupported module subclass {type(module).__name__}: "
                    "custom forward semantics are not lowered"
                )

    supported = {
        torch.float32: "float32",
        torch.float64: "float64",
    }
    model_dtypes: set[object] = set()
    for tensor_name, tensor in _trusted_named_tensors(modules, torch):
        if tensor.dtype not in supported:
            raise UnsupportedOpError(
                f"Unsupported dtype {tensor.dtype} for tensor {tensor_name}; "
                "only float32 and float64 are supported"
            )
        model_dtypes.add(tensor.dtype)
    if len(model_dtypes) > 1:
        names = ", ".join(sorted(str(dtype) for dtype in model_dtypes))
        raise UnsupportedOpError(
            f"Unsupported mixed model dtypes: {names}; use one floating dtype"
        )

    requested_dtype = _normalize_float_dtype(input_dtype, torch, supported)
    model_dtype = next(iter(model_dtypes), None)
    if requested_dtype is not None and model_dtype is not None and requested_dtype != model_dtype:
        raise UnsupportedOpError(
            f"Unsupported input dtype {requested_dtype} for model dtype {model_dtype}"
        )
    resolved = requested_dtype or model_dtype or torch.get_default_dtype()
    if resolved not in supported:
        raise UnsupportedOpError(
            f"Unsupported dtype {resolved}; only float32 and float64 are supported"
        )
    for module_name, module in modules:
        _validate_no_custom_introspection(module, module_name or "<root>", nn)
    return supported[resolved]


def _validate_module_dispatch(
    module: object,
    display_name: str,
    nn: object,
) -> None:
    module_type = type(module)
    state = _raw_module_state(module)
    for attribute in (
        "__call__",
        "_wrapped_call_impl",
        "_call_impl",
        "_compiled_call_impl",
    ):
        if inspect.getattr_static(module_type, attribute) is not inspect.getattr_static(
            nn.Module,
            attribute,
        ):
            label = "compiled call" if attribute == "_compiled_call_impl" else "call"
            raise UnsupportedOpError(
                f"Unsupported custom {label} path ({attribute}) on module "
                f"{display_name}: call semantics are not lowered"
            )
        if attribute in state:
            raise UnsupportedOpError(
                f"Unsupported instance call override ({attribute}) on module "
                f"{display_name}: call semantics are not lowered"
            )
    for attribute in ("__getattribute__", "__getattr__"):
        if inspect.getattr_static(module_type, attribute) is not inspect.getattr_static(
            nn.Module,
            attribute,
        ):
            raise UnsupportedOpError(
                f"Unsupported custom call path ({attribute}) on module "
                f"{display_name}: call semantics are not lowered"
            )
    if "forward" in state:
        raise UnsupportedOpError(
            f"Unsupported instance forward override on module {display_name}"
        )


def _validate_no_custom_introspection(
    module: object,
    display_name: str,
    nn: object,
) -> None:
    state = _raw_module_state(module)
    for attribute in (
        "named_modules",
        "named_parameters",
        "named_buffers",
        "named_children",
        "modules",
        "parameters",
        "buffers",
        "children",
    ):
        if inspect.getattr_static(type(module), attribute) is not inspect.getattr_static(
            nn.Module,
            attribute,
        ) or attribute in state:
            raise UnsupportedOpError(
                f"Unsupported module introspection override ({attribute}) on "
                f"module {display_name}"
            )


def _copy_model_for_tracing(model: object, nn: object, torch: object) -> object:
    before_state = _snapshot_module_state(model, nn, torch)
    class_state = _snapshot_module_class_definitions(model, nn)
    try:
        copied = copy.deepcopy(model)
    except Exception as exc:
        _restore_module_class_definitions(class_state)
        raise UnsupportedOpError(
            f"Unsupported model state: model cannot be copied safely for FX trace: {exc}"
        ) from exc
    class_changed = _restore_module_class_definitions(class_state)
    original_changed = before_state != _snapshot_module_state(model, nn, torch)
    if class_changed or original_changed:
        raise UnsupportedOpError(
            "Unsupported model copy: deepcopy mutated Python module state"
        )
    if not isinstance(copied, nn.Module):
        raise UnsupportedOpError(
            "Unsupported model copy: deepcopy did not return an nn.Module"
        )
    if before_state != _snapshot_module_state(copied, nn, torch):
        raise UnsupportedOpError(
            "Unsupported model copy: deepcopy changed or substituted module semantics"
        )
    return copied


def _validate_no_custom_copy_protocol(module_type: type[object], nn: object) -> None:
    protocol_names = {
        "__deepcopy__",
        "__reduce__",
        "__reduce_ex__",
        "__getstate__",
        "__setstate__",
        "__getnewargs__",
        "__getnewargs_ex__",
    }
    framework_classes = _framework_module_classes(nn)
    for cls in module_type.__mro__:
        if cls in framework_classes:
            break
        for name in protocol_names & vars(cls).keys():
            raise UnsupportedOpError(
                f"Unsupported custom model copy protocol {cls.__name__}.{name}"
            )


def _validate_no_custom_class_state(root_type: type[object], nn: object) -> None:
    metadata_names = {"__module__", "__qualname__", "__doc__"}
    framework_classes = _framework_module_classes(nn)
    for cls in root_type.__mro__:
        if cls in framework_classes:
            break
        for name, value in vars(cls).items():
            if name in metadata_names:
                continue
            if name == "__annotations__" and type(value) is dict and not value:
                continue
            if type(value) in (staticmethod, classmethod):
                continue
            if isinstance(value, FunctionType):
                if name == "__init__" or not (
                    name.startswith("__") and name.endswith("__")
                ):
                    continue
                raise UnsupportedOpError(
                    f"Unsupported custom class descriptor {cls.__name__}.{name}"
                )
            raise UnsupportedOpError(
                f"Unsupported custom class-level state or descriptor "
                f"{cls.__name__}.{name}"
            )


def _validate_forward_python_state(
    module_type: type[object],
    nn: object,
    torch: object,
) -> None:
    framework_classes = _framework_module_classes(nn)
    for cls in module_type.__mro__:
        if cls in framework_classes:
            break
        forward = vars(cls).get("forward")
        if forward is None:
            continue
        _validate_reachable_python_function(
            function=forward,
            label=f"{cls.__name__}.forward",
            module_type=module_type,
            nn=nn,
            torch=torch,
            seen=set(),
        )


def _validate_reachable_python_function(
    function: object,
    label: str,
    module_type: type[object],
    nn: object,
    torch: object,
    seen: set[int],
) -> None:
    if not isinstance(function, FunctionType):
        raise UnsupportedOpError(f"Unsupported callable descriptor in {label}")
    identity = id(function)
    if identity in seen:
        return
    seen.add(identity)
    if getattr(function, "__wrapped__", None) is not None or inspect.unwrap(
        function
    ) is not function:
        raise UnsupportedOpError(f"Unsupported decorated function {label}")
    if function.__dict__:
        raise UnsupportedOpError(f"Unsupported custom function state in {label}")
    if function.__builtins__ is not _TRUSTED_BUILTINS:
        raise UnsupportedOpError(f"Unsupported custom builtins mapping in {label}")
    if builtins.len is not _TRUSTED_LEN:
        raise UnsupportedOpError(f"Unsupported modified builtins binding in {label}")
    if any(_contains_code_object(value) for value in function.__code__.co_consts):
        raise UnsupportedOpError(f"Unsupported nested code object in {label}")
    if any(
        not _is_immutable_python_state(value, set())
        for value in function.__code__.co_consts
    ):
        raise UnsupportedOpError(f"Unsupported code constant in {label}")
    defaults = (
        *tuple(function.__defaults__ or ()),
        *tuple((function.__kwdefaults__ or {}).values()),
    )
    if any(not _is_immutable_python_state(value, set()) for value in defaults):
        raise UnsupportedOpError(f"Unsupported external Python state in {label} defaults")
    if function.__closure__ is not None:
        for cell in function.__closure__:
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if not _is_immutable_python_state(value, set()):
                raise UnsupportedOpError(f"Unsupported external Python state in {label}")
    bytecode = dis.Bytecode(function)
    if bytecode.exception_entries:
        raise UnsupportedOpError(f"Unsupported exception handling in {label}")
    instructions = tuple(bytecode)
    for instruction in instructions:
        if instruction.opname == "FORMAT_VALUE":
            raise UnsupportedOpError(
                f"Unsupported tensor-dependent string formatting in {label}"
            )
        if instruction.opname in {"STORE_GLOBAL", "DELETE_GLOBAL"}:
            raise UnsupportedOpError(f"Unsupported external Python state in {label}")
        if instruction.opname not in {"LOAD_ATTR", "LOAD_METHOD"}:
            continue
        method_name = str(instruction.argval)
        if method_name in {"format", "node", "tracer"}:
            raise UnsupportedOpError(
                f"Unsupported Python introspection/string formatting attribute "
                f"{method_name!r} in {label}"
            )
        if method_name in {"_parameters", "_buffers", "_modules"}:
            raise UnsupportedOpError(
                f"Unsupported internal module registry access {method_name!r} in {label}"
            )
        if method_name.startswith("__") and method_name.endswith("__"):
            raise UnsupportedOpError(
                f"Unsupported Python introspection attribute {method_name!r} in {label}"
            )
    if any(
        instruction.opcode in dis.hasjabs or instruction.opcode in dis.hasjrel
        for instruction in instructions
    ):
        raise UnsupportedOpError(
            "Unsupported FX symbolic trace control flow/state semantics in "
            f"{label}"
        )
    supported_opcodes = {
        "BINARY_OP",
        "BUILD_TUPLE",
        "CACHE",
        "CALL",
        "COPY_FREE_VARS",
        "EXTENDED_ARG",
        "KW_NAMES",
        "LOAD_ATTR",
        "LOAD_CONST",
        "LOAD_DEREF",
        "LOAD_FAST",
        "LOAD_GLOBAL",
        "LOAD_METHOD",
        "NOP",
        "POP_TOP",
        "PRECALL",
        "PUSH_NULL",
        "RESUME",
        "RETURN_VALUE",
        "STORE_FAST",
    }
    for index, instruction in enumerate(instructions):
        if instruction.opname == "FORMAT_VALUE":
            raise UnsupportedOpError(
                f"Unsupported tensor-dependent string formatting in {label}"
            )
        if instruction.opname == "BINARY_OP" and instruction.argrepr == "%":
            raise UnsupportedOpError(
                f"Unsupported tensor-dependent string formatting in {label}"
            )
        if instruction.opname in {"STORE_GLOBAL", "DELETE_GLOBAL"}:
            raise UnsupportedOpError(f"Unsupported external Python state in {label}")
        if instruction.opname not in supported_opcodes:
            raise UnsupportedOpError(
                f"Unsupported custom Python bytecode {instruction.opname} in {label}"
            )
        if instruction.opname == "LOAD_GLOBAL":
            name = str(instruction.argval)
            if name not in function.__globals__:
                if name == "len":
                    continue
                raise UnsupportedOpError(
                    f"Unsupported external Python state {name!r} in {label}"
                )
            value = function.__globals__[name]
            if value is torch and _next_loaded_attribute(instructions, index) == "argmax":
                continue
            raise UnsupportedOpError(
                f"Unsupported external Python state {name!r} in {label}"
            )
        if instruction.opname not in {"LOAD_ATTR", "LOAD_METHOD"}:
            continue
        method_name = str(instruction.argval)
        if method_name in {"format", "node", "tracer"}:
            raise UnsupportedOpError(
                f"Unsupported Python introspection/string formatting attribute "
                f"{method_name!r} in {label}"
            )
        if method_name in {"_parameters", "_buffers", "_modules"}:
            raise UnsupportedOpError(
                f"Unsupported internal module registry access {method_name!r} in {label}"
            )
        if method_name.startswith("__") and method_name.endswith("__"):
            raise UnsupportedOpError(
                f"Unsupported Python introspection attribute {method_name!r} in {label}"
            )
        resolved = _find_custom_method(module_type, method_name, nn)
        if resolved is not None:
            owner, method = resolved
            _validate_reachable_python_function(
                function=method,
                label=f"{owner.__name__}.{method_name}",
                module_type=module_type,
                nn=nn,
                torch=torch,
                seen=seen,
            )


def _find_custom_method(
    module_type: type[object],
    name: str,
    nn: object,
) -> tuple[type[object], object] | None:
    framework_classes = _framework_module_classes(nn)
    for cls in module_type.__mro__:
        if cls in framework_classes:
            break
        if name in vars(cls):
            value = vars(cls)[name]
            if isinstance(value, FunctionType):
                return cls, value
            if type(value) in (staticmethod, classmethod):
                return cls, value.__func__
            return None
    return None


def _framework_module_classes(nn: object) -> frozenset[type[object]]:
    current_types = (
        nn.Module,
        nn.Sequential,
        nn.Identity,
        nn.Linear,
        nn.Conv2d,
        nn.ReLU,
        nn.Flatten,
    )
    if any(
        current is not trusted
        for current, trusted in zip(
            current_types,
            _TRUSTED_FRAMEWORK_TYPES,
            strict=True,
        )
    ):
        raise UnsupportedOpError("Unsupported modified PyTorch framework types")
    return _TRUSTED_FRAMEWORK_CLASSES


def _validate_framework_integrity(nn: object) -> None:
    if (
        copy.deepcopy is not _TRUSTED_DEEPCOPY
        or copy.deepcopy.__code__ is not _TRUSTED_DEEPCOPY_CODE
    ):
        raise UnsupportedOpError("Unsupported modified model copy function")
    for owner, name, function, fingerprint in _TRUSTED_FUNCTION_FINGERPRINTS:
        current = inspect.getattr_static(owner, name)
        if current is not function or _definition_fingerprint(current) != fingerprint:
            raise UnsupportedOpError(
                f"Unsupported modified PyTorch framework function {name}"
            )
    _framework_module_classes(nn)
    if nn.Parameter is not _TRUSTED_PARAMETER_TYPE:
        raise UnsupportedOpError("Unsupported modified PyTorch framework Parameter")
    for cls, before in _TRUSTED_FRAMEWORK_DEFINITIONS:
        current = dict(vars(cls))
        if current.keys() != before.keys() or any(
            current.get(name) is not value
            or _definition_fingerprint(current.get(name)) != fingerprint
            for name, (value, fingerprint) in before.items()
        ):
            raise UnsupportedOpError(
                f"Unsupported modified PyTorch framework class {cls.__name__}"
            )


def _next_loaded_attribute(
    instructions: tuple[dis.Instruction, ...],
    index: int,
) -> str | None:
    for following in instructions[index + 1 :]:
        if following.opname in {"CACHE", "EXTENDED_ARG", "NOP"}:
            continue
        if following.opname in {"LOAD_ATTR", "LOAD_METHOD"}:
            return str(following.argval)
        return None
    return None


def _is_immutable_python_state(value: object, seen: set[int]) -> bool:
    if value is None or type(value) in (bool, int, float, complex, str, bytes):
        return True
    identity = id(value)
    if identity in seen:
        return True
    seen.add(identity)
    try:
        if type(value) is tuple:
            return all(_is_immutable_python_state(item, seen) for item in value)
        if type(value) is frozenset:
            return all(_is_immutable_python_state(item, seen) for item in value)
        return False
    finally:
        seen.remove(identity)


def _contains_code_object(value: object) -> bool:
    if isinstance(value, CodeType):
        return True
    if type(value) in (tuple, frozenset):
        return any(_contains_code_object(item) for item in value)
    return False


def _validate_copy_safe_instance_state(
    module: object,
    display_name: str,
    torch: object,
) -> None:
    for name, value in _raw_module_state(module).items():
        if name in {"_modules", "_parameters", "_buffers"}:
            continue
        if not _is_copy_safe_state_value(value, torch, set()):
            raise UnsupportedOpError(
                f"Unsupported external Python state {name!r} on module "
                f"{display_name}: value cannot be copied safely"
            )


def _is_copy_safe_state_value(value: object, torch: object, seen: set[int]) -> bool:
    if value is None or type(value) in (bool, int, float, complex, str, bytes):
        return True
    if type(value) in (torch.Tensor, torch.nn.Parameter, np.ndarray):
        return False
    if type(value) in (torch.dtype, torch.device):
        return True
    identity = id(value)
    if identity in seen:
        return True
    seen.add(identity)
    try:
        if type(value) in (dict, OrderedDict):
            return all(
                _is_copy_safe_state_value(key, torch, seen)
                and _is_copy_safe_state_value(item, torch, seen)
                for key, item in value.items()
            )
        if type(value) in (list, tuple, set, frozenset):
            return all(_is_copy_safe_state_value(item, torch, seen) for item in value)
        return False
    finally:
        seen.remove(identity)


def _snapshot_module_class_definitions(
    model: object,
    nn: object,
) -> tuple[tuple[type[object], dict[str, object]], ...]:
    snapshots: list[tuple[type[object], dict[str, object]]] = []
    seen: set[type[object]] = set()
    framework_classes = _framework_module_classes(nn)
    for _, module in _trusted_named_modules(model, nn):
        for cls in type(module).__mro__:
            if cls in framework_classes:
                break
            if cls not in seen:
                snapshots.append((cls, dict(vars(cls))))
                seen.add(cls)
    return tuple(snapshots)


def _restore_module_class_definitions(
    snapshots: tuple[tuple[type[object], dict[str, object]], ...],
) -> bool:
    changed = False
    for cls, before in snapshots:
        current = dict(vars(cls))
        if current.keys() != before.keys() or any(
            current.get(name) is not value for name, value in before.items()
        ):
            changed = True
        for name in current.keys() - before.keys():
            delattr(cls, name)
        for name, value in before.items():
            if current.get(name) is not value:
                setattr(cls, name, value)
    return changed


def _raw_module_state(module: object) -> dict[str, object]:
    state = object.__getattribute__(module, "__dict__")
    if type(state) is not dict:
        raise UnsupportedOpError("Unsupported nn.Module state mapping")
    if any(type(name) is not str for name in state):
        raise UnsupportedOpError("Unsupported non-string nn.Module state key")
    return state


def _trusted_named_modules(model: object, nn: object) -> tuple[tuple[str, object], ...]:
    modules: list[tuple[str, object]] = []
    seen: set[int] = set()

    def visit(name: str, module: object) -> None:
        if not isinstance(module, nn.Module):
            raise UnsupportedOpError(
                f"Unsupported module tree entry {name or '<root>'}: expected nn.Module"
            )
        identity = id(module)
        if identity in seen:
            return
        seen.add(identity)
        modules.append((name, module))
        children = _raw_module_state(module).get("_modules")
        if type(children) is not dict:
            raise UnsupportedOpError(
                f"Unsupported module tree on {name or '<root>'}: _modules must be a dict"
            )
        for child_name, child in children.items():
            if type(child_name) is not str:
                raise UnsupportedOpError("Unsupported non-string module name")
            if child is None:
                continue
            qualified = f"{name}.{child_name}" if name else child_name
            visit(qualified, child)

    visit("", model)
    return tuple(modules)


def _trusted_named_tensors(
    modules: tuple[tuple[str, object], ...],
    torch: object,
) -> tuple[tuple[str, object], ...]:
    tensors: list[tuple[str, object]] = []
    for module_name, module in modules:
        state = _raw_module_state(module)
        for registry_name in ("_parameters", "_buffers"):
            registry = state.get(registry_name)
            if type(registry) is not dict:
                raise UnsupportedOpError(
                    f"Unsupported module tensor registry {registry_name} on "
                    f"{module_name or '<root>'}"
                )
            for tensor_name, tensor in registry.items():
                if type(tensor_name) is not str:
                    raise UnsupportedOpError(
                        f"Unsupported non-string tensor name in {registry_name}"
                    )
                if tensor is None:
                    continue
                if not isinstance(tensor, torch.Tensor):
                    raise UnsupportedOpError(
                        f"Unsupported non-tensor value in {registry_name} on "
                        f"{module_name or '<root>'}"
                    )
                expected_type = (
                    torch.nn.Parameter
                    if registry_name == "_parameters"
                    else torch.Tensor
                )
                if type(tensor) is not expected_type:
                    raise UnsupportedOpError(
                        f"Unsupported tensor subclass {type(tensor).__name__} in "
                        f"{registry_name} on {module_name or '<root>'}"
                    )
                if tensor.device.type != "cpu":
                    raise UnsupportedOpError(
                        f"Unsupported tensor device {tensor.device} for "
                        f"{module_name or '<root>'}.{tensor_name}; only CPU is supported"
                    )
                if tensor.layout is not torch.strided:
                    raise UnsupportedOpError(
                        f"Unsupported tensor layout {tensor.layout} for "
                        f"{module_name or '<root>'}.{tensor_name}; only strided is supported"
                    )
                if not tensor.is_contiguous():
                    raise UnsupportedOpError(
                        f"Unsupported non-contiguous tensor "
                        f"{module_name or '<root>'}.{tensor_name}"
                    )
                if tensor.is_conj():
                    raise UnsupportedOpError(
                        f"Unsupported conjugate view bit on tensor "
                        f"{module_name or '<root>'}.{tensor_name}"
                    )
                if tensor.is_neg():
                    raise UnsupportedOpError(
                        f"Unsupported negative view bit on tensor "
                        f"{module_name or '<root>'}.{tensor_name}"
                    )
                if tensor.grad is not None:
                    raise UnsupportedOpError(
                        f"Unsupported gradient state on tensor "
                        f"{module_name or '<root>'}.{tensor_name}; "
                        "autograd state is not lowered"
                    )
                qualified = f"{module_name}.{tensor_name}" if module_name else tensor_name
                tensors.append((qualified, tensor))
    return tuple(tensors)


def _snapshot_module_state(
    model: object,
    nn: object,
    torch: object,
) -> tuple[object, ...]:
    snapshots: list[object] = []
    for module_name, module in _trusted_named_modules(model, nn):
        state = _raw_module_state(module)
        attributes = tuple(
            sorted(
                (
                    name,
                    _freeze_state_value(value, torch, set()),
                )
                for name, value in state.items()
                if name != "_modules"
            )
        )
        child_registry = state["_modules"]
        children = tuple(
            (name, type(child))
            for name, child in child_registry.items()
            if child is not None
        )
        snapshots.append(
            (
                module_name,
                type(module),
                attributes,
                children,
            )
        )
    return tuple(snapshots)


def _freeze_state_value(value: object, torch: object, seen: set[int]) -> object:
    if value is None or type(value) in (bool, int, float, complex, str, bytes):
        return (type(value).__qualname__, repr(value))
    if type(value) in (torch.Tensor, torch.nn.Parameter):
        data = (
            value.detach()
            .resolve_conj()
            .resolve_neg()
            .cpu()
            .contiguous()
            .numpy()
        )
        gradient = None
        if value.grad is not None:
            gradient = _freeze_state_value(value.grad, torch, seen)
        return (
            "tensor",
            type(value).__module__,
            type(value).__qualname__,
            str(value.dtype),
            str(value.device),
            str(value.layout),
            tuple(value.shape),
            tuple(value.stride()),
            value.storage_offset(),
            value.requires_grad,
            value.is_conj(),
            value.is_neg(),
            gradient,
            data.tobytes(),
        )
    if type(value) is np.ndarray:
        data = np.ascontiguousarray(value)
        return (
            "ndarray",
            str(value.dtype),
            tuple(value.shape),
            tuple(value.strides),
            value.flags.c_contiguous,
            value.flags.f_contiguous,
            value.flags.owndata,
            value.flags.writeable,
            value.flags.aligned,
            data.tobytes(),
        )

    identity = id(value)
    if identity in seen:
        return ("cycle", identity)
    seen.add(identity)
    try:
        if isinstance(value, dict):
            items = [
                (
                    _freeze_state_value(key, torch, seen),
                    _freeze_state_value(item, torch, seen),
                )
                for key, item in value.items()
            ]
            return ("dict", tuple(sorted(items, key=repr)))
        if isinstance(value, (list, tuple)):
            return (
                type(value).__qualname__,
                tuple(_freeze_state_value(item, torch, seen) for item in value),
            )
        if isinstance(value, (set, frozenset)):
            items = [_freeze_state_value(item, torch, seen) for item in value]
            return (type(value).__qualname__, tuple(sorted(items, key=repr)))
        if callable(value):
            return (
                "callable",
                getattr(value, "__module__", ""),
                getattr(value, "__qualname__", repr(value)),
            )
        state = getattr(value, "__dict__", None)
        if isinstance(state, dict):
            return (
                "object",
                type(value).__module__,
                type(value).__qualname__,
                _freeze_state_value(state, torch, seen),
            )
        return ("value", type(value).__module__, type(value).__qualname__, repr(value))
    finally:
        seen.remove(identity)


def _validate_lowered_semantics(
    model: object,
    graph: GraphIR,
    input_shape: tuple[int, ...],
    float_dtype: str,
    nn: object,
    torch: object,
) -> None:
    from torch2rtl.quant.reference import infer_float_graph

    dtype = torch.float32 if float_dtype == "float32" else torch.float64
    numpy_dtype = np.dtype(float_dtype)
    element_count = math.prod(input_shape)
    probe_values = (
        np.zeros(input_shape, dtype=numpy_dtype),
        np.linspace(-0.75, 0.75, num=element_count, dtype=numpy_dtype).reshape(
            input_shape
        ),
    )
    for values in probe_values:
        probe_model = _copy_model_for_tracing(model, nn, torch)
        before_state = _snapshot_module_state(probe_model, nn, torch)
        class_state = _snapshot_module_class_definitions(probe_model, nn)
        probe_error: Exception | None = None
        actual: object | None = None
        probe_input = torch.from_numpy(values.copy()).to(dtype=dtype)
        try:
            with torch.no_grad():
                actual = probe_model(probe_input)
        except Exception as exc:
            probe_error = exc
        class_changed = _restore_module_class_definitions(class_state)
        instance_changed = before_state != _snapshot_module_state(
            probe_model,
            nn,
            torch,
        )
        if class_changed or instance_changed:
            raise UnsupportedOpError(
                "Unsupported Python state mutation during concrete semantic probe"
            )
        if probe_error is not None:
            raise UnsupportedOpError(
                f"Unsupported concrete model semantics: {probe_error}"
            ) from probe_error
        if not isinstance(actual, torch.Tensor):
            raise UnsupportedOpError(
                "Unsupported model output: concrete forward must return one Tensor"
            )

        expected_dtype = torch.int64 if graph.output.dtype == "int64" else dtype
        if actual.dtype != expected_dtype:
            raise UnsupportedOpError(
                "Unsupported model output dtype: concrete forward produced "
                f"{actual.dtype}, GraphIR requires {expected_dtype}"
            )
        actual_values = actual.detach().cpu().numpy()
        expected_values = infer_float_graph(graph, values).output
        if actual_values.shape != expected_values.shape or not np.array_equal(
            actual_values,
            expected_values,
            equal_nan=True,
        ):
            raise UnsupportedOpError(
                "Unsupported model semantics: concrete forward does not match lowered GraphIR"
            )


def _normalize_float_dtype(
    value: object | None,
    torch: object,
    supported: dict[object, str],
) -> object | None:
    if value is None:
        return None
    if value is torch.float32 or value is np.dtype(np.float32):
        return torch.float32
    if value is torch.float64 or value is np.dtype(np.float64):
        return torch.float64
    if type(value) is str:
        if value in {"float32", "torch.float32"}:
            return torch.float32
        if value in {"float64", "torch.float64"}:
            return torch.float64
    raise UnsupportedOpError(
        "Unsupported input dtype; only float32 and float64 are supported"
    )


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


def _load_python_module(path: Path) -> ModuleType:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load Python module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _safe_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name)
    if cleaned and cleaned[0].isdigit():
        cleaned = f"op_{cleaned}"
    return cleaned or "op"
