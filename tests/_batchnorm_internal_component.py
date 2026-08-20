from __future__ import annotations

import builtins
import copy
import dis
import inspect
import math
import pickle
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
import torch.nn as nn

import torch2rtl.frontend.pytorch_fx as fx_frontend
from torch2rtl.frontend.pytorch_fx import UnsupportedOpError, parse_model
from torch2rtl.ir.ops import Conv2dIR, ReluIR
from torch2rtl.quant.reference import infer_float_graph

from _batchnorm_bootstrap_runtime import (
    _assert_bootstrap_rejection_preserves_source_and_clean_parse,
)


def _basic_model(
    *,
    in_channels: int = 1,
    out_channels: int = 1,
    conv_bias: bool = True,
    bn_affine: bool = True,
    dtype: torch.dtype = torch.float32,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=conv_bias,
            dtype=dtype,
        ),
        nn.BatchNorm2d(
            out_channels,
            affine=bn_affine,
            track_running_stats=True,
            dtype=dtype,
        ),
    ).eval()


def _tensor_snapshot(tensor: torch.Tensor) -> tuple[object, ...]:
    values = (
        tensor.detach()
        .resolve_conj()
        .resolve_neg()
        .cpu()
        .contiguous()
        .numpy()
    )
    gradient = None
    if tensor.grad is not None:
        gradient = _tensor_snapshot(tensor.grad)
    return (
        id(tensor),
        type(tensor),
        str(tensor.dtype),
        str(tensor.device),
        str(tensor.layout),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.requires_grad,
        tensor.is_conj(),
        tensor.is_neg(),
        gradient,
        values.tobytes(),
    )


def _source_snapshot(model: nn.Module) -> tuple[object, ...]:
    modules = tuple(
        (
            name,
            id(module),
            type(module),
            module.training,
            id(type(module).forward),
            tuple((key, id(value)) for key, value in module._forward_hooks.items()),
            tuple(
                (key, id(value)) for key, value in module._forward_pre_hooks.items()
            ),
            tuple(
                (key, id(value)) for key, value in module._backward_hooks.items()
            ),
        )
        for name, module in model.named_modules(remove_duplicate=False)
    )
    parameters = tuple(
        (name, _tensor_snapshot(parameter))
        for name, parameter in model.named_parameters(remove_duplicate=False)
    )
    buffers = tuple(
        (name, _tensor_snapshot(buffer))
        for name, buffer in model.named_buffers(remove_duplicate=False)
    )
    marker = getattr(model, "marker", None)
    return modules, parameters, buffers, marker



@pytest.mark.parametrize(
    "binding_name",
    ["_parse_model_impl", "_validate_framework_integrity_impl"],
)
def test_parser_and_validator_implementations_reject_replace_then_delete(
    binding_name: str,
) -> None:
    model = _basic_model()
    before = _source_snapshot(model)
    original = fx_frontend.__dict__[binding_name]
    callback_calls = 0
    attribute_calls = 0

    class HostileImplementation:
        def __getattribute__(self, name: str) -> object:
            nonlocal attribute_calls
            attribute_calls += 1
            raise AssertionError(
                f"untrusted implementation attribute accessed: {binding_name}.{name}"
            )

        def __call__(self, *_args: object, **_kwargs: object) -> object:
            nonlocal callback_calls
            callback_calls += 1
            raise AssertionError(f"untrusted implementation called: {binding_name}")

    replacement = HostileImplementation()
    caught_errors: list[BaseException | None] = []
    setattr(fx_frontend, binding_name, replacement)
    try:
        caught: BaseException | None = None
        try:
            parse_model(model, input_shape=(1, 2, 2))
        except BaseException as exc:
            caught = exc
        caught_errors.append(caught)
    finally:
        setattr(fx_frontend, binding_name, original)

    delattr(fx_frontend, binding_name)
    try:
        caught = None
        try:
            parse_model(model, input_shape=(1, 2, 2))
        except BaseException as exc:
            caught = exc
        caught_errors.append(caught)
    finally:
        setattr(fx_frontend, binding_name, original)

    assert callback_calls == 0
    assert attribute_calls == 0
    expected_label = (
        "parser implementation"
        if binding_name == "_parse_model_impl"
        else "integrity implementation"
    )
    assert len(caught_errors) == 2
    for caught in caught_errors:
        assert expected_label in str(caught).lower()
        _assert_bootstrap_rejection_preserves_source_and_clean_parse(
            model,
            before,
            caught,
        )


@pytest.mark.parametrize("mutation", ["replace", "delete"])
def test_stable_parse_reference_rejects_modified_public_binding(
    mutation: str,
) -> None:
    model = _basic_model()
    before = _source_snapshot(model)
    stable_parse = parse_model
    original = fx_frontend.parse_model
    calls = 0

    def hostile_parse(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("untrusted public parse binding must not execute")

    if mutation == "replace":
        fx_frontend.parse_model = hostile_parse
    else:
        delattr(fx_frontend, "parse_model")
    caught: BaseException | None = None
    try:
        try:
            stable_parse(model, input_shape=(1, 2, 2))
        except BaseException as exc:
            caught = exc
    finally:
        fx_frontend.parse_model = original

    assert calls == 0
    assert type(caught) is UnsupportedOpError
    assert "parser binding" in str(caught).lower()
    assert _source_snapshot(model) == before

    graph = fx_frontend.parse_model(model, input_shape=(1, 2, 2))

    assert [type(op) for op in graph.ops] == [Conv2dIR]
    assert _source_snapshot(model) == before


def test_public_parse_wrapper_has_exact_stable_signature() -> None:
    signature = inspect.signature(fx_frontend.parse_model)

    assert str(signature) == (
        "(model: 'object', input_shape: 'Sequence[int]', "
        "input_dtype: 'object | None' = None) -> 'GraphIR'"
    )
    assert fx_frontend.parse_model.__name__ == "parse_model"
    assert fx_frontend.parse_model.__qualname__ == "parse_model"
    assert fx_frontend.parse_model.__module__ == fx_frontend.__name__


def test_public_parse_wrapper_pickle_round_trip_preserves_identity_and_behavior() -> None:
    restored = pickle.loads(pickle.dumps(fx_frontend.parse_model))
    model = _basic_model()
    before = _source_snapshot(model)

    assert restored is fx_frontend.parse_model
    graph = restored(model, input_shape=(1, 2, 2))
    assert [type(op) for op in graph.ops] == [Conv2dIR]
    assert _source_snapshot(model) == before


@pytest.mark.parametrize("mutation", ["replace", "delete"])
@pytest.mark.parametrize(
    "helper_name",
    [
        "_class_local_fingerprint",
        "_definition_fingerprint",
        "_descriptor_local_fingerprint",
        "_framework_module_classes",
        "_function_local_fingerprint",
        "_function_referenced_globals_fingerprint",
        "_raw_python_module_namespace",
        "_raw_static_attribute",
        "_require_exact_string_dict",
        "_shallow_state_fingerprint",
    ],
)
def test_rejects_replaced_or_deleted_frontend_integrity_helper_before_it_runs(
    helper_name: str,
    mutation: str,
) -> None:
    model = _basic_model()
    before = _source_snapshot(model)
    original = getattr(fx_frontend, helper_name)
    calls = 0

    def hostile_helper(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError(f"untrusted helper must not execute: {helper_name}")

    if mutation == "replace":
        setattr(fx_frontend, helper_name, hostile_helper)
    else:
        delattr(fx_frontend, helper_name)
    caught: BaseException | None = None
    try:
        try:
            parse_model(model, input_shape=(1, 2, 2))
        except BaseException as exc:
            caught = exc
    finally:
        setattr(fx_frontend, helper_name, original)

    assert calls == 0
    assert helper_name in str(caught)
    _assert_bootstrap_rejection_preserves_source_and_clean_parse(
        model,
        before,
        caught,
    )


def test_missing_nn_module_does_not_invoke_hostile_module_getattr() -> None:
    model = _basic_model()
    before = _source_snapshot(model)
    namespace = nn.__dict__
    original_module = namespace["Module"]
    missing = object()
    original_getattr = namespace.get("__getattr__", missing)
    calls = 0

    def hostile_getattr(name: str) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError(f"untrusted torch.nn.__getattr__ called for {name}")

    delattr(nn, "Module")
    setattr(nn, "__getattr__", hostile_getattr)
    caught: BaseException | None = None
    try:
        try:
            parse_model(model, input_shape=(1, 2, 2))
        except BaseException as exc:
            caught = exc
    finally:
        setattr(nn, "Module", original_module)
        if original_getattr is missing:
            delattr(nn, "__getattr__")
        else:
            setattr(nn, "__getattr__", original_getattr)

    assert calls == 0
    assert "framework" in str(caught).lower() or "module" in str(caught).lower()
    _assert_bootstrap_rejection_preserves_source_and_clean_parse(
        model,
        before,
        caught,
    )


@pytest.mark.parametrize("hostile_kind", ["class-property", "metaclass-equality"])
def test_protected_fusion_global_rejects_hostile_type_protocol_without_calling_it(
    hostile_kind: str,
) -> None:
    model = _basic_model()
    before = _source_snapshot(model)
    function = torch.nn.utils.fusion.fuse_conv_bn_eval
    function_globals = function.__globals__
    original = function_globals["copy"]
    calls = 0

    if hostile_kind == "class-property":
        class HostileValue:
            @property
            def __class__(self) -> type[object]:
                nonlocal calls
                calls += 1
                raise AssertionError("untrusted __class__ property must not execute")

        replacement: object = HostileValue()
    else:
        class HostileMeta(type):
            def __eq__(cls, other: object) -> bool:
                nonlocal calls
                calls += 1
                raise AssertionError("untrusted metaclass equality must not execute")

        class HostileValue(metaclass=HostileMeta):
            pass

        replacement = HostileValue

    function_globals["copy"] = replacement
    caught: BaseException | None = None
    try:
        try:
            parse_model(model, input_shape=(1, 2, 2))
        except BaseException as exc:
            caught = exc
    finally:
        function_globals["copy"] = original

    assert calls == 0
    _assert_bootstrap_rejection_preserves_source_and_clean_parse(
        model,
        before,
        caught,
    )


def test_protected_dis_cache_rejects_hostile_metaclass_hash_without_calling_it() -> None:
    model = _basic_model()
    before = _source_snapshot(model)
    cache_format = dis._cache_format
    key = "__torch2rtl_hostile_hash__"
    assert key not in cache_format
    calls = 0

    class HostileMeta(type):
        def __hash__(cls) -> int:
            nonlocal calls
            calls += 1
            raise AssertionError("untrusted metaclass hash must not execute")

    class HostileValue(metaclass=HostileMeta):
        pass

    cache_format[key] = HostileValue
    caught: BaseException | None = None
    try:
        try:
            parse_model(model, input_shape=(1, 2, 2))
        except BaseException as exc:
            caught = exc
    finally:
        del cache_format[key]

    assert calls == 0
    _assert_bootstrap_rejection_preserves_source_and_clean_parse(
        model,
        before,
        caught,
    )


def test_colliding_builtin_key_and_deleted_type_fail_without_equality_call() -> None:
    model = _basic_model()
    before = _source_snapshot(model)
    original_type = builtins.type
    collision_hash = hash("type")
    equality_calls = 0

    class CollidingKey:
        def __hash__(self) -> int:
            return collision_hash

        def __eq__(self, other: object) -> bool:
            nonlocal equality_calls
            equality_calls += 1
            raise AssertionError("untrusted builtins key equality must not execute")

    hostile_key = CollidingKey()
    del builtins.type
    builtins.__dict__[hostile_key] = None
    equality_calls = 0
    caught: BaseException | None = None
    try:
        try:
            parse_model(model, input_shape=(1, 2, 2))
        except BaseException as exc:
            caught = exc
    finally:
        del builtins.__dict__[hostile_key]
        builtins.type = original_type

    assert equality_calls == 0
    _assert_bootstrap_rejection_preserves_source_and_clean_parse(
        model,
        before,
        caught,
    )


def test_colliding_protected_function_global_key_fails_without_equality_call() -> None:
    model = _basic_model()
    before = _source_snapshot(model)
    function_globals = torch.nn.utils.fusion.fuse_conv_bn_eval.__globals__
    original_copy = function_globals.pop("copy")
    collision_hash = hash("copy")
    equality_calls = 0

    class CollidingKey:
        def __hash__(self) -> int:
            return collision_hash

        def __eq__(self, other: object) -> bool:
            nonlocal equality_calls
            equality_calls += 1
            raise AssertionError("untrusted globals key equality must not execute")

    hostile_key = CollidingKey()
    function_globals[hostile_key] = None
    equality_calls = 0
    caught: BaseException | None = None
    try:
        try:
            parse_model(model, input_shape=(1, 2, 2))
        except BaseException as exc:
            caught = exc
    finally:
        del function_globals[hostile_key]
        function_globals["copy"] = original_copy

    assert equality_calls == 0
    _assert_bootstrap_rejection_preserves_source_and_clean_parse(
        model,
        before,
        caught,
    )


def test_colliding_dis_constructor_global_key_fails_without_equality_call() -> None:
    model = _basic_model()
    before = _source_snapshot(model)
    descriptor = type.__getattribute__(dis.Positions, "__dict__")["__new__"]
    constructor_globals = descriptor.__func__.__globals__
    original_tuple_new = constructor_globals.pop("_tuple_new")
    collision_hash = hash("_tuple_new")
    equality_calls = 0

    class CollidingKey:
        def __hash__(self) -> int:
            return collision_hash

        def __eq__(self, other: object) -> bool:
            nonlocal equality_calls
            equality_calls += 1
            raise AssertionError(
                "untrusted dis constructor key equality must not execute"
            )

    hostile_key = CollidingKey()
    constructor_globals[hostile_key] = None
    equality_calls = 0
    caught: BaseException | None = None
    try:
        try:
            parse_model(model, input_shape=(1, 2, 2))
        except BaseException as exc:
            caught = exc
    finally:
        del constructor_globals[hostile_key]
        constructor_globals["_tuple_new"] = original_tuple_new

    assert equality_calls == 0
    _assert_bootstrap_rejection_preserves_source_and_clean_parse(
        model,
        before,
        caught,
    )


def test_rejects_replaced_builtin_len_without_calling_it() -> None:
    model = _basic_model()
    called = 0
    original = builtins.len

    def hostile_len(_value: object) -> int:
        nonlocal called
        called += 1
        raise AssertionError("untrusted builtins.len must not execute")

    builtins.len = hostile_len
    called = 0
    caught: UnsupportedOpError | None = None
    try:
        try:
            parse_model(model, input_shape=(1, 2, 2))
        except UnsupportedOpError as exc:
            caught = exc
    finally:
        builtins.len = original

    assert caught is not None
    assert "builtins" in str(caught).lower()
    assert called == 0
