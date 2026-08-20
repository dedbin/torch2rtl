from __future__ import annotations

import copy
from collections import OrderedDict
from types import FunctionType

import numpy as np

from torch2rtl.frontend._errors import UnsupportedOpError


def _state_framework_module_classes(nn: object) -> frozenset[type[object]]:
    module_types = (
        nn.Module,
        nn.Sequential,
        nn.Identity,
        nn.Linear,
        nn.Conv2d,
        nn.BatchNorm2d,
        nn.ReLU,
        nn.Flatten,
    )
    return frozenset(cls for module_type in module_types for cls in module_type.__mro__)

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
    framework_classes = _state_framework_module_classes(nn)
    for cls in module_type.__mro__:
        if cls in framework_classes:
            break
        for name in protocol_names & vars(cls).keys():
            raise UnsupportedOpError(
                f"Unsupported custom model copy protocol {cls.__name__}.{name}"
            )


def _validate_no_custom_class_state(root_type: type[object], nn: object) -> None:
    metadata_names = {"__module__", "__qualname__", "__doc__"}
    framework_classes = _state_framework_module_classes(nn)
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
    framework_classes = _state_framework_module_classes(nn)
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
) -> tuple[tuple[str, object, bool], ...]:
    tensors: list[tuple[str, object, bool]] = []
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
                is_batchnorm_counter = (
                    type(module) is torch.nn.BatchNorm2d
                    and registry_name == "_buffers"
                    and tensor_name == "num_batches_tracked"
                )
                if is_batchnorm_counter and (
                    tensor.dtype is not torch.int64 or tuple(tensor.shape) != ()
                ):
                    raise UnsupportedOpError(
                        "Unsupported BatchNorm2d num_batches_tracked: expected "
                        "a scalar int64 Tensor"
                    )
                tensors.append((qualified, tensor, is_batchnorm_counter))
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
