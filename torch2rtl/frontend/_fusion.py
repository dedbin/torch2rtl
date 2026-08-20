from __future__ import annotations

import math
from types import FunctionType

import numpy as np

from torch2rtl.frontend._errors import UnsupportedOpError
from torch2rtl.frontend._state_guard import _raw_module_state, _trusted_named_modules


def _fusion_require_exact_string_dict(
    value: object,
    label: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise UnsupportedOpError(f"Unsupported modified {label} mapping")
    for key in value:
        if type(key) is not str:
            raise UnsupportedOpError(f"Unsupported modified {label} keys")
    return value


def _fusion_raw_static_attribute(owner: object, name: str) -> object:
    namespace = object.__getattribute__(owner, "__dict__")
    if type(namespace) is dict:
        _fusion_require_exact_string_dict(namespace, "runtime namespace")
        if name in namespace:
            return namespace[name]
        raise AttributeError(name)
    for cls in type.__getattribute__(owner, "__mro__"):
        namespace = type.__getattribute__(cls, "__dict__")
        if name in namespace:
            return namespace[name]
    raise AttributeError(name)


def _fusion_is_fx_node(value: object) -> bool:
    return (
        hasattr(value, "op")
        and hasattr(value, "target")
        and hasattr(value, "args")
        and hasattr(value, "kwargs")
    )


def _fusion_required_module_attribute(
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


def _fusion_positive_module_integer(
    value: object,
    module: str,
    attribute: str,
) -> int:
    if type(value) is not int:
        raise UnsupportedOpError(
            f"Unsupported {module} {attribute}: expected a built-in integer, "
            f"got {value!r}"
        )
    if value <= 0:
        raise UnsupportedOpError(
            f"Unsupported {module} {attribute}: expected a positive built-in integer, "
            f"got {value!r}"
        )
    return value


def _fusion_required_parameter(
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


def _fusion_optional_parameter(
    value: object,
    module: str,
    attribute: str,
    nn: object,
) -> object | None:
    if value is None:
        return None
    return _fusion_required_parameter(value, module, attribute, nn)


def _fuse_conv_batchnorm_eval(
    traced: object,
    nn: object,
    torch: object,
) -> tuple[object, list[dict[str, str]]]:
    modules = dict(_trusted_named_modules(traced, nn))
    candidates: list[tuple[object, object, object, object]] = []
    for node in tuple(traced.graph.nodes):
        if node.op != "call_module":
            continue
        module = modules.get(str(node.target))
        if type(module) is not nn.BatchNorm2d:
            continue
        args = tuple(node.args)
        kwargs = dict(node.kwargs)
        if len(args) != 1 or kwargs or not _fusion_is_fx_node(args[0]):
            raise UnsupportedOpError(
                f"Unsupported BatchNorm2d call at node {node.name}: expected "
                "one direct Conv2d tensor input"
            )
        conv_node = args[0]
        conv = (
            modules.get(str(conv_node.target))
            if conv_node.op == "call_module"
            else None
        )
        if type(conv) is not nn.Conv2d:
            raise UnsupportedOpError(
                f"Unsupported BatchNorm2d at node {node.name}: only an exact "
                "Conv2d -> BatchNorm2d pattern can be fused"
            )
        if _fx_node_use_count(traced.graph, conv_node) != 1:
            raise UnsupportedOpError(
                f"Unsupported Conv2d fan-out at node {conv_node.name}: "
                "fusion requires exactly one FX edge use"
            )
        if _fx_node_use_count(traced.graph, node) != 1:
            raise UnsupportedOpError(
                f"Unsupported BatchNorm2d fan-out at node {node.name}: "
                "fusion requires exactly one FX edge use"
            )
        candidates.append((conv_node, conv, node, module))

    if not candidates:
        return traced, []

    _require_eval_module(traced, "root model")
    transformations: list[dict[str, str]] = []
    for index, (conv_node, conv, batchnorm_node, batchnorm) in enumerate(candidates):
        _validate_conv_batchnorm_pair(conv, batchnorm, nn, torch)
        try:
            fused = torch.nn.utils.fusion.fuse_conv_bn_eval(conv, batchnorm)
        except (AssertionError, RuntimeError, TypeError, ValueError) as exc:
            raise UnsupportedOpError(
                "Unsupported Conv2d -> BatchNorm2d fusion at nodes "
                f"{conv_node.name} -> {batchnorm_node.name}: {exc}"
            ) from exc
        _validate_fused_conv(fused, conv, nn)
        fused_target = _fresh_fused_target(traced, index)
        traced.add_module(fused_target, fused)
        original_conv_target = str(conv_node.target)
        original_batchnorm_target = str(batchnorm_node.target)
        conv_node.target = fused_target
        batchnorm_node.replace_all_uses_with(conv_node)
        traced.graph.erase_node(batchnorm_node)
        transformations.append(
            {
                "kind": "conv2d_batchnorm2d_fusion",
                "conv_node": str(conv_node.name),
                "conv_target": original_conv_target,
                "batchnorm_node": str(batchnorm_node.name),
                "batchnorm_target": original_batchnorm_target,
                "fused_target": fused_target,
            }
        )

    traced.graph.lint()
    traced.recompile()
    traced.delete_all_unused_submodules()
    traced.graph.lint()
    traced.recompile()
    remaining_modules = dict(_trusted_named_modules(traced, nn))
    for node in traced.graph.nodes:
        if node.op == "call_module" and type(
            remaining_modules.get(str(node.target))
        ) is nn.BatchNorm2d:
            raise UnsupportedOpError(
                f"Unsupported unfused BatchNorm2d at FX node {node.name}"
            )
    return traced, transformations


def _validate_conv_batchnorm_pair(
    conv: object,
    batchnorm: object,
    nn: object,
    torch: object,
) -> None:
    if type(conv) is not nn.Conv2d or type(batchnorm) is not nn.BatchNorm2d:
        raise UnsupportedOpError(
            "Unsupported fusion module types: exact Conv2d and BatchNorm2d "
            "instances are required"
        )
    conv_state = _raw_module_state(conv)
    if {"weight", "bias"}.intersection(conv_state):
        raise UnsupportedOpError(
            "Unsupported Conv2d instance schema: parameter slots must not be "
            "shadowed by direct attributes"
        )
    _require_eval_module(conv, "Conv2d")
    _require_eval_module(batchnorm, "BatchNorm2d")

    out_channels = _fusion_positive_module_integer(
        _fusion_required_module_attribute(conv, "Conv2d", "out_channels"),
        "Conv2d",
        "out_channels",
    )
    num_features = _fusion_positive_module_integer(
        _fusion_required_module_attribute(batchnorm, "BatchNorm2d", "num_features"),
        "BatchNorm2d",
        "num_features",
    )
    if num_features != out_channels:
        raise UnsupportedOpError(
            "Unsupported BatchNorm2d num_features: expected Conv2d "
            f"out_channels={out_channels}, got {num_features}"
        )

    state = _raw_module_state(batchnorm)
    parameters = state.get("_parameters")
    buffers = state.get("_buffers")
    children = state.get("_modules")
    registry_names = {
        "weight",
        "bias",
        "running_mean",
        "running_var",
        "num_batches_tracked",
    }
    shadowed_names = registry_names.intersection(state)
    if shadowed_names:
        raise UnsupportedOpError(
            "Unsupported BatchNorm2d instance schema: parameter and buffer "
            "slots must not be shadowed by direct attributes"
        )
    if type(children) is not dict or children:
        raise UnsupportedOpError(
            "Unsupported BatchNorm2d module schema: child modules are not allowed"
        )
    if type(parameters) is not dict or set(parameters) != {"weight", "bias"}:
        raise UnsupportedOpError(
            "Unsupported BatchNorm2d parameter schema: expected exactly "
            "weight and bias slots"
        )
    expected_buffers = {"running_mean", "running_var", "num_batches_tracked"}
    if type(buffers) is not dict or set(buffers) != expected_buffers:
        raise UnsupportedOpError(
            "Unsupported BatchNorm2d buffer schema: expected exactly "
            "running_mean, running_var, and num_batches_tracked slots"
        )
    non_persistent = state.get("_non_persistent_buffers_set")
    if type(non_persistent) is not set or non_persistent:
        raise UnsupportedOpError(
            "Unsupported BatchNorm2d buffer schema: running statistics must "
            "be persistent"
        )

    affine = _fusion_required_module_attribute(batchnorm, "BatchNorm2d", "affine")
    if type(affine) is not bool:
        raise UnsupportedOpError(
            "Unsupported BatchNorm2d affine: expected a built-in boolean"
        )
    weight = parameters["weight"]
    bias = parameters["bias"]
    if affine:
        weight = _fusion_required_parameter(weight, "BatchNorm2d", "weight", nn)
        try:
            constructor = _fusion_raw_static_attribute(nn.BatchNorm2d, "__init__")
        except AttributeError as exc:
            raise UnsupportedOpError(
                "Unsupported BatchNorm2d constructor metadata"
            ) from exc
        if type(constructor) is not FunctionType:
            raise UnsupportedOpError(
                "Unsupported BatchNorm2d constructor metadata"
            )
        constructor_code = constructor.__code__
        kwonly_start = constructor_code.co_argcount
        kwonly_stop = kwonly_start + constructor_code.co_kwonlyargcount
        kwdefaults = constructor.__kwdefaults__
        supports_optional_bias = (
            "bias" in constructor_code.co_varnames[kwonly_start:kwonly_stop]
            and type(kwdefaults) is dict
            and kwdefaults.get("bias") is True
        )
        if supports_optional_bias:
            bias = _fusion_optional_parameter(bias, "BatchNorm2d", "bias", nn)
        else:
            bias = _fusion_required_parameter(bias, "BatchNorm2d", "bias", nn)
    elif weight is not None or bias is not None:
        raise UnsupportedOpError(
            "Unsupported BatchNorm2d parameter schema: affine=False requires "
            "empty weight and bias slots"
        )

    track_running_stats = _fusion_required_module_attribute(
        batchnorm,
        "BatchNorm2d",
        "track_running_stats",
    )
    if type(track_running_stats) is not bool or not track_running_stats:
        raise UnsupportedOpError(
            "Unsupported BatchNorm2d track_running_stats: running statistics "
            "are required for eval fusion"
        )
    running_mean = buffers["running_mean"]
    running_var = buffers["running_var"]
    counter = buffers["num_batches_tracked"]
    for name, tensor in (
        ("running_mean", running_mean),
        ("running_var", running_var),
    ):
        if type(tensor) is not torch.Tensor:
            raise UnsupportedOpError(
                f"Unsupported BatchNorm2d {name}: expected a Tensor"
            )
        if tuple(tensor.shape) != (num_features,):
            raise UnsupportedOpError(
                f"Unsupported BatchNorm2d {name} shape: expected "
                f"({num_features},), got {tuple(tensor.shape)}"
            )
        if tensor.requires_grad:
            raise UnsupportedOpError(
                f"Unsupported BatchNorm2d {name}: buffers must not require gradients"
            )
    if (
        type(counter) is not torch.Tensor
        or counter.dtype is not torch.int64
        or tuple(counter.shape) != ()
        or counter.requires_grad
    ):
        raise UnsupportedOpError(
            "Unsupported BatchNorm2d num_batches_tracked: expected a "
            "non-gradient scalar int64 Tensor"
        )
    counter_value = int(counter.detach().cpu().numpy().item())
    if counter_value < 0:
        raise UnsupportedOpError(
            "Unsupported BatchNorm2d num_batches_tracked: expected a "
            "non-negative value"
        )

    expected_shape = (num_features,)
    for name, tensor in (("weight", weight), ("bias", bias)):
        if tensor is not None and tuple(tensor.shape) != expected_shape:
            raise UnsupportedOpError(
                f"Unsupported BatchNorm2d {name} shape: expected "
                f"{expected_shape}, got {tuple(tensor.shape)}"
            )

    conv_weight = _fusion_required_parameter(
        _fusion_required_module_attribute(conv, "Conv2d", "weight"),
        "Conv2d",
        "weight",
        nn,
    )
    conv_bias = _fusion_optional_parameter(
        _fusion_required_module_attribute(conv, "Conv2d", "bias"),
        "Conv2d",
        "bias",
        nn,
    )
    floating_tensors = [conv_weight, running_mean, running_var]
    floating_tensors.extend(
        tensor for tensor in (conv_bias, weight, bias) if tensor is not None
    )
    dtype = conv_weight.dtype
    if dtype not in (torch.float32, torch.float64) or any(
        tensor.dtype is not dtype for tensor in floating_tensors
    ):
        raise UnsupportedOpError(
            "Unsupported Conv2d/BatchNorm2d dtype: all fusion parameters and "
            "statistics must use one float32 or float64 dtype"
        )
    for name, tensor in (
        ("Conv2d weight", conv_weight),
        ("Conv2d bias", conv_bias),
        ("BatchNorm2d weight", weight),
        ("BatchNorm2d bias", bias),
        ("BatchNorm2d running_mean", running_mean),
        ("BatchNorm2d running_var", running_var),
    ):
        if tensor is not None and not _tensor_values_are_finite(tensor):
            raise UnsupportedOpError(
                f"Unsupported {name}: fusion inputs must contain only finite values"
            )

    eps = _fusion_required_module_attribute(batchnorm, "BatchNorm2d", "eps")
    if type(eps) is not float or not math.isfinite(eps) or eps < 0.0:
        raise UnsupportedOpError(
            "Unsupported BatchNorm2d eps: expected a finite non-negative "
            "built-in float"
        )
    running_var_values = running_var.detach().cpu().numpy()
    if np.any(running_var_values < 0):
        raise UnsupportedOpError(
            "Unsupported BatchNorm2d running_var: values must be non-negative"
        )
    variance_with_eps = (running_var + eps).detach().cpu().numpy()
    if not np.all(np.isfinite(variance_with_eps)) or np.any(variance_with_eps <= 0):
        raise UnsupportedOpError(
            "Unsupported BatchNorm2d running_var + eps: every channel must "
            "have a finite positive denominator"
        )


def _require_eval_module(module: object, label: str) -> None:
    training = _fusion_required_module_attribute(module, label, "training")
    if type(training) is not bool or training:
        raise UnsupportedOpError(
            f"Unsupported {label} training state: Conv2d/BatchNorm2d fusion "
            "requires eval mode"
        )


def _validate_fused_conv(
    fused: object,
    source_conv: object,
    nn: object,
) -> None:
    if type(fused) is not nn.Conv2d:
        raise UnsupportedOpError(
            "Unsupported official fusion result: expected an exact Conv2d"
        )
    weight = _fusion_required_parameter(
        _fusion_required_module_attribute(fused, "fused Conv2d", "weight"),
        "fused Conv2d",
        "weight",
        nn,
    )
    bias = _fusion_required_parameter(
        _fusion_required_module_attribute(fused, "fused Conv2d", "bias"),
        "fused Conv2d",
        "bias",
        nn,
    )
    source_weight = _fusion_required_parameter(
        _fusion_required_module_attribute(source_conv, "Conv2d", "weight"),
        "Conv2d",
        "weight",
        nn,
    )
    if weight.dtype is not source_weight.dtype or bias.dtype is not source_weight.dtype:
        raise UnsupportedOpError(
            "Unsupported official fusion result: fused parameter dtype changed"
        )
    if not _tensor_values_are_finite(weight) or not _tensor_values_are_finite(bias):
        raise UnsupportedOpError(
            "Unsupported official fusion result: fused Conv2d parameters "
            "must contain only finite values"
        )
    _require_eval_module(fused, "fused Conv2d")


def _tensor_values_are_finite(tensor: object) -> bool:
    values = tensor.detach().cpu().numpy()
    return bool(np.all(np.isfinite(values)))


def _fresh_fused_target(traced: object, index: int) -> str:
    base = f"_torch2rtl_fused_conv_bn_{index}"
    candidate = base
    suffix = 0
    state = _raw_module_state(traced)
    modules = state.get("_modules")
    if type(modules) is not dict:
        raise UnsupportedOpError("Unsupported FX GraphModule module registry")
    while candidate in state or candidate in modules or hasattr(traced, candidate):
        suffix += 1
        candidate = f"{base}_{suffix}"
    return candidate


def _fx_node_use_count(graph: object, target: object) -> int:
    count = 0
    for node in graph.nodes:
        count += _fx_value_reference_count(node.args, target)
        count += _fx_value_reference_count(node.kwargs, target)
    return count


def _fx_value_reference_count(value: object, target: object) -> int:
    if value is target:
        return 1
    if isinstance(value, (tuple, list)):
        count = 0
        for item in value:
            count += _fx_value_reference_count(item, target)
        return count
    if isinstance(value, dict):
        count = 0
        for key, item in value.items():
            count += _fx_value_reference_count(key, target)
            count += _fx_value_reference_count(item, target)
        return count
    if type(value) is slice:
        count = 0
        for item in (value.start, value.stop, value.step):
            count += _fx_value_reference_count(item, target)
        return count
    return 0
