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



def _assert_bootstrap_rejection_preserves_source_and_clean_parse(
    model: nn.Sequential,
    before: tuple[object, ...],
    caught: BaseException | None,
) -> None:
    assert type(caught) is UnsupportedOpError
    message = str(caught).lower()
    assert any(
        label in message
        for label in (
            "builtins",
            "dis",
            "framework",
            "function globals",
            "integrity",
            "binding",
            "module binding",
            "parser",
            "runtime",
            "moduletype",
            "trust root",
        )
    )
    assert _source_snapshot(model) == before

    graph = parse_model(model, input_shape=(1, 2, 2))

    assert [type(op) for op in graph.ops] == [Conv2dIR]
    assert _source_snapshot(model) == before


@pytest.mark.parametrize("replacement_mode", ["side-effect", "throwing"])
def test_rejects_replaced_builtin_type_before_it_runs_and_restores_cleanly(
    replacement_mode: str,
) -> None:
    model = _basic_model()
    before = _source_snapshot(model)
    original = builtins.type
    calls = 0

    def replacement(value: object) -> object:
        nonlocal calls
        calls += 1
        if replacement_mode == "throwing":
            raise AssertionError("untrusted builtins.type must not execute")
        return original(value)

    builtins.type = replacement
    caught: BaseException | None = None
    try:
        try:
            parse_model(model, input_shape=(1, 2, 2))
        except BaseException as exc:
            caught = exc
    finally:
        builtins.type = original

    assert calls == 0
    _assert_bootstrap_rejection_preserves_source_and_clean_parse(
        model,
        before,
        caught,
    )


def test_rejects_deleted_builtin_type_and_restores_cleanly() -> None:
    model = _basic_model()
    before = _source_snapshot(model)
    original = builtins.type
    del builtins.type
    caught: BaseException | None = None
    try:
        try:
            parse_model(model, input_shape=(1, 2, 2))
        except BaseException as exc:
            caught = exc
    finally:
        builtins.type = original

    _assert_bootstrap_rejection_preserves_source_and_clean_parse(
        model,
        before,
        caught,
    )


def test_sequential_builtin_type_patches_restore_before_clean_parse() -> None:
    model = _basic_model()
    before = _source_snapshot(model)
    original = builtins.type
    calls = 0

    def tracking_type(value: object) -> type[object]:
        nonlocal calls
        calls += 1
        return original(value)

    caught_errors: list[BaseException | None] = []
    builtins.type = tracking_type
    try:
        caught: BaseException | None = None
        try:
            parse_model(model, input_shape=(1, 2, 2))
        except BaseException as exc:
            caught = exc
        caught_errors.append(caught)
    finally:
        builtins.type = original

    del builtins.type
    try:
        caught = None
        try:
            parse_model(model, input_shape=(1, 2, 2))
        except BaseException as exc:
            caught = exc
        caught_errors.append(caught)
    finally:
        builtins.type = original

    assert calls == 0
    assert len(caught_errors) == 2
    for caught in caught_errors:
        assert type(caught) is UnsupportedOpError
    _assert_bootstrap_rejection_preserves_source_and_clean_parse(
        model,
        before,
        caught_errors[-1],
    )


def test_rejects_deleted_builtin_dict_and_restores_cleanly() -> None:
    model = _basic_model()
    before = _source_snapshot(model)
    original = builtins.dict
    del builtins.dict
    caught: BaseException | None = None
    try:
        try:
            parse_model(model, input_shape=(1, 2, 2))
        except BaseException as exc:
            caught = exc
    finally:
        builtins.dict = original

    _assert_bootstrap_rejection_preserves_source_and_clean_parse(
        model,
        before,
        caught,
    )


@pytest.mark.parametrize("mutation", ["replace", "delete"])
@pytest.mark.parametrize(
    "binding_name",
    ["ModuleType", "builtins", "copy", "dis", "inspect", "math", "np", "sys"],
)
def test_rejects_replaced_or_deleted_bootstrap_global_before_it_runs(
    binding_name: str,
    mutation: str,
) -> None:
    model = _basic_model()
    before = _source_snapshot(model)
    original = fx_frontend.__dict__[binding_name]
    calls = 0

    class HostileBinding:
        def __getattribute__(self, name: str) -> object:
            nonlocal calls
            calls += 1
            raise AssertionError(
                f"untrusted bootstrap binding accessed: {binding_name}.{name}"
            )

    if mutation == "replace":
        setattr(fx_frontend, binding_name, HostileBinding())
    else:
        delattr(fx_frontend, binding_name)
    caught: BaseException | None = None
    try:
        try:
            parse_model(model, input_shape=(1, 2, 2))
        except BaseException as exc:
            caught = exc
    finally:
        setattr(fx_frontend, binding_name, original)

    assert calls == 0
    _assert_bootstrap_rejection_preserves_source_and_clean_parse(
        model,
        before,
        caught,
    )


@pytest.mark.parametrize("mutation", ["replace", "delete"])
@pytest.mark.parametrize(
    ("owner_name", "binding_name"),
    [
        pytest.param("builtins", "staticmethod", id="builtins-staticmethod"),
        pytest.param("builtins", "classmethod", id="builtins-classmethod"),
        pytest.param("builtins", "property", id="builtins-property"),
        pytest.param("frontend", "FunctionType", id="frontend-function-type"),
        pytest.param("frontend", "CodeType", id="frontend-code-type"),
    ],
)
def test_rejects_replaced_or_deleted_bootstrap_type_binding_before_it_runs(
    owner_name: str,
    binding_name: str,
    mutation: str,
) -> None:
    model = _basic_model()
    before = _source_snapshot(model)
    owner = builtins if owner_name == "builtins" else fx_frontend
    original = owner.__dict__[binding_name]
    calls = 0

    def hostile_binding(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError(f"untrusted binding must not run: {binding_name}")

    if mutation == "replace":
        setattr(owner, binding_name, hostile_binding)
    else:
        delattr(owner, binding_name)
    caught: BaseException | None = None
    try:
        try:
            parse_model(model, input_shape=(1, 2, 2))
        except BaseException as exc:
            caught = exc
    finally:
        setattr(owner, binding_name, original)

    assert calls == 0
    assert binding_name.lower() in str(caught).lower()
    _assert_bootstrap_rejection_preserves_source_and_clean_parse(
        model,
        before,
        caught,
    )


@pytest.mark.parametrize("mutation", ["replace", "delete"])
@pytest.mark.parametrize(
    "binding_name",
    ["_validate_framework_integrity", "UnsupportedOpError"],
)
def test_rejects_replaced_or_deleted_validator_entry_binding_before_it_runs(
    binding_name: str,
    mutation: str,
) -> None:
    model = _basic_model()
    before = _source_snapshot(model)
    original = fx_frontend.__dict__[binding_name]
    calls = 0

    def hostile_binding(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError(f"untrusted validator binding ran: {binding_name}")

    if mutation == "replace":
        setattr(fx_frontend, binding_name, hostile_binding)
    else:
        delattr(fx_frontend, binding_name)
    caught: BaseException | None = None
    try:
        try:
            parse_model(model, input_shape=(1, 2, 2))
        except BaseException as exc:
            caught = exc
    finally:
        setattr(fx_frontend, binding_name, original)

    assert calls == 0
    expected_label = (
        "integrity validator"
        if binding_name == "_validate_framework_integrity"
        else "error binding"
    )
    assert expected_label in str(caught).lower()
    _assert_bootstrap_rejection_preserves_source_and_clean_parse(
        model,
        before,
        caught,
    )


_CURRENT_TRUSTED_GLOBAL_NAMES = tuple(
    sorted(
        name
        for name in fx_frontend.__dict__
        if name.startswith("_TRUSTED_")
    )
)
_REPORTED_MUTABLE_TRUST_ROOTS = frozenset(
    {
        "_TRUSTED_FRONTEND_GLOBALS",
        "_TRUSTED_LEN",
        "_TRUSTED_MODULE_GETATTRIBUTE",
        "_TRUSTED_TORCH_MODULE",
        "_TRUSTED_TORCH_NN_MODULE",
        "_TRUSTED_TYPE",
        "_TRUSTED_VALIDATE_FRAMEWORK_INTEGRITY",
    }
)


def test_trusted_global_matrix_covers_every_current_and_reported_root() -> None:
    current = frozenset(
        name
        for name in fx_frontend.__dict__
        if name.startswith("_TRUSTED_")
    )

    assert frozenset(_CURRENT_TRUSTED_GLOBAL_NAMES) == current
    assert _REPORTED_MUTABLE_TRUST_ROOTS <= current


@pytest.mark.parametrize("trusted_name", _CURRENT_TRUSTED_GLOBAL_NAMES)
def test_every_trusted_global_rejects_replace_then_delete_without_running_it(
    trusted_name: str,
) -> None:
    model = _basic_model()
    before = _source_snapshot(model)
    original = fx_frontend.__dict__[trusted_name]
    callback_calls = 0
    attribute_calls = 0

    class HostileTrustedRoot:
        def __getattribute__(self, name: str) -> object:
            nonlocal attribute_calls
            attribute_calls += 1
            raise AssertionError(
                f"untrusted trusted-root attribute accessed: {trusted_name}.{name}"
            )

        def __call__(self, *_args: object, **_kwargs: object) -> object:
            nonlocal callback_calls
            callback_calls += 1
            raise AssertionError(f"untrusted trusted root called: {trusted_name}")

    replacement = HostileTrustedRoot()
    setattr(fx_frontend, trusted_name, replacement)
    replaced_error: BaseException | None = None
    try:
        try:
            parse_model(model, input_shape=(1, 2, 2))
        except BaseException as exc:
            replaced_error = exc
    finally:
        setattr(fx_frontend, trusted_name, original)

    assert callback_calls == 0
    assert attribute_calls == 0
    if trusted_name == "_TRUSTED_VALIDATE_FRAMEWORK_INTEGRITY":
        assert "validator trust root" in str(replaced_error).lower()
    else:
        assert trusted_name in str(replaced_error)
    _assert_bootstrap_rejection_preserves_source_and_clean_parse(
        model,
        before,
        replaced_error,
    )

    delattr(fx_frontend, trusted_name)
    deleted_error: BaseException | None = None
    try:
        try:
            parse_model(model, input_shape=(1, 2, 2))
        except BaseException as exc:
            deleted_error = exc
    finally:
        setattr(fx_frontend, trusted_name, original)

    assert callback_calls == 0
    assert attribute_calls == 0
    if trusted_name == "_TRUSTED_VALIDATE_FRAMEWORK_INTEGRITY":
        assert "validator trust root" in str(deleted_error).lower()
    else:
        assert trusted_name in str(deleted_error)
    _assert_bootstrap_rejection_preserves_source_and_clean_parse(
        model,
        before,
        deleted_error,
    )
