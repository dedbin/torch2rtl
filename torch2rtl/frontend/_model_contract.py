from __future__ import annotations

import builtins
import dis
from types import CodeType, FunctionType

import numpy as np

from torch2rtl.frontend._errors import UnsupportedOpError
from torch2rtl.frontend._state_guard import (
    _raw_module_state,
    _state_framework_module_classes,
    _trusted_named_modules,
    _trusted_named_tensors,
    _validate_copy_safe_instance_state,
    _validate_no_custom_class_state,
    _validate_no_custom_copy_protocol,
)


_BUILTINS = builtins.__dict__
_LEN = len
_DIS_BYTECODE = dis.Bytecode
_DIS_HASJABS = dis.hasjabs
_DIS_HASJREL = dis.hasjrel


def _contract_require_exact_string_dict(
    value: object,
    label: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise UnsupportedOpError(f"Unsupported modified {label} mapping")
    for key in value:
        if type(key) is not str:
            raise UnsupportedOpError(f"Unsupported modified {label} keys")
    return value


def _contract_raw_static_attribute(owner: object, name: str) -> object:
    namespace = object.__getattribute__(owner, "__dict__")
    if type(namespace) is dict:
        _contract_require_exact_string_dict(namespace, "runtime namespace")
        if name in namespace:
            return namespace[name]
        raise AttributeError(name)
    for cls in type.__getattribute__(owner, "__mro__"):
        namespace = type.__getattribute__(cls, "__dict__")
        if name in namespace:
            return namespace[name]
    raise AttributeError(name)


def _validate_model_semantics(
    model: object,
    nn: object,
    torch: object,
    input_dtype: object | None,
) -> str:
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

    lowered_types = (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Flatten)
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
        if state.get("_backward_hooks") or state.get("_backward_pre_hooks"):
            raise UnsupportedOpError(
                f"Unsupported backward hook on module {display_name}: "
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
    for tensor_name, tensor, is_batchnorm_counter in _trusted_named_tensors(
        modules,
        torch,
    ):
        if is_batchnorm_counter:
            continue
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
        if _contract_raw_static_attribute(module_type, attribute) is not _contract_raw_static_attribute(
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
        if _contract_raw_static_attribute(module_type, attribute) is not _contract_raw_static_attribute(
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
        if _contract_raw_static_attribute(type(module), attribute) is not _contract_raw_static_attribute(
            nn.Module,
            attribute,
        ) or attribute in state:
            raise UnsupportedOpError(
                f"Unsupported module introspection override ({attribute}) on "
                f"module {display_name}"
            )


def _validate_forward_python_state(
    module_type: type[object],
    nn: object,
    torch: object,
) -> None:
    framework_classes = _state_framework_module_classes(nn)
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
    if "__wrapped__" in function.__dict__:
        raise UnsupportedOpError(f"Unsupported decorated function {label}")
    if function.__dict__:
        raise UnsupportedOpError(f"Unsupported custom function state in {label}")
    if function.__builtins__ is not _BUILTINS:
        raise UnsupportedOpError(f"Unsupported custom builtins mapping in {label}")
    if builtins.len is not _LEN:
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
    framework_classes = _state_framework_module_classes(nn)
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
