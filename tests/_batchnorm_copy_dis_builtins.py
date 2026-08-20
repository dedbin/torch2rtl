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



@pytest.mark.parametrize(
    "dispatch_name",
    ["_deepcopy_dispatch", "dispatch_table"],
)
def test_rejects_modified_copy_dispatch_before_untrusted_handler_runs(
    dispatch_name: str,
) -> None:
    model = _basic_model()
    dispatch = getattr(copy, dispatch_name)
    called = 0

    def hostile_handler(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called += 1
        raise AssertionError("untrusted copy dispatch handler must not execute")

    key = dict if dispatch_name == "_deepcopy_dispatch" else nn.Sequential
    missing = object()
    original = dispatch.get(key, missing)
    dispatch[key] = hostile_handler
    caught: UnsupportedOpError | None = None
    try:
        try:
            parse_model(model, input_shape=(1, 2, 2))
        except UnsupportedOpError as exc:
            caught = exc
    finally:
        if original is missing:
            del dispatch[key]
        else:
            dispatch[key] = original

    assert caught is not None
    assert "copy" in str(caught).lower()
    assert called == 0


def test_restored_copy_dispatch_order_does_not_poison_later_parse() -> None:
    model = nn.Sequential(nn.Conv2d(1, 1, 1), nn.ReLU()).eval()
    dispatch = copy._deepcopy_dispatch
    original_items = tuple(dispatch.items())
    key, expected_value = original_items[0]
    removed_value = dispatch.pop(key)
    assert removed_value is expected_value
    dispatch[key] = removed_value
    try:
        graph = parse_model(model, input_shape=(1, 2, 2))
    finally:
        dispatch.clear()
        dispatch.update(original_items)

    assert [type(op) for op in graph.ops] == [Conv2dIR, ReluIR]


def test_preimport_copyreg_entry_with_custom_metaclass_remains_supported() -> None:
    script = """
import copyreg
import torch.nn as nn

class Meta(type):
    pass

class Registered(metaclass=Meta):
    pass

def reduce_registered(value):
    return Registered, ()

copyreg.pickle(Registered, reduce_registered)
from torch2rtl.frontend.pytorch_fx import parse_model

graph = parse_model(
    nn.Sequential(nn.Conv2d(1, 1, 1), nn.ReLU()).eval(),
    input_shape=(1, 2, 2),
)
assert len(graph.ops) == 2
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_restored_dis_class_binding_order_does_not_poison_later_parse() -> None:
    model = nn.Sequential(nn.Conv2d(1, 1, 1), nn.ReLU()).eval()
    cls = dis.Bytecode
    original = type.__getattribute__(cls, "__dict__")["__iter__"]
    delattr(cls, "__iter__")
    setattr(cls, "__iter__", original)

    graph = parse_model(model, input_shape=(1, 2, 2))

    assert [type(op) for op in graph.ops] == [Conv2dIR, ReluIR]


def test_rejects_replaced_copy_helper_before_untrusted_helper_runs() -> None:
    model = _basic_model()
    called = 0
    original = copy._reconstruct

    def hostile_reconstruct(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called += 1
        raise AssertionError("untrusted copy helper must not execute")

    copy._reconstruct = hostile_reconstruct
    caught: UnsupportedOpError | None = None
    try:
        try:
            parse_model(model, input_shape=(1, 2, 2))
        except UnsupportedOpError as exc:
            caught = exc
    finally:
        copy._reconstruct = original

    assert caught is not None
    assert "copy" in str(caught).lower()
    assert called == 0


def test_rejects_modified_deepcopy_defaults_before_untrusted_memo_runs() -> None:
    model = _basic_model()
    called = 0

    class HostileMemo(dict[object, object]):
        def get(self, *_args: object, **_kwargs: object) -> object:
            nonlocal called
            called += 1
            raise AssertionError("untrusted deepcopy memo must not execute")

    original = copy.deepcopy.__defaults__
    copy.deepcopy.__defaults__ = (HostileMemo(), [])
    caught: UnsupportedOpError | None = None
    try:
        try:
            parse_model(model, input_shape=(1, 2, 2))
        except UnsupportedOpError as exc:
            caught = exc
    finally:
        copy.deepcopy.__defaults__ = original

    assert caught is not None
    assert "copy" in str(caught).lower()
    assert called == 0


def test_rejects_hostile_dis_jump_entry_without_executing_equality() -> None:
    if not dis.hasjrel:
        pytest.skip("runtime has no relative-jump opcode table entries")
    model = _basic_model()
    equality_calls = 0

    class HostileInt(int):
        def __eq__(self, other: object) -> bool:
            nonlocal equality_calls
            equality_calls += 1
            raise AssertionError("untrusted jump-table equality must not execute")

    original = dis.hasjrel[0]
    dis.hasjrel[0] = HostileInt(original)
    caught: UnsupportedOpError | None = None
    try:
        try:
            parse_model(model, input_shape=(1, 2, 2))
        except UnsupportedOpError as exc:
            caught = exc
    finally:
        dis.hasjrel[0] = original

    assert caught is not None
    assert "dis" in str(caught).lower()
    assert equality_calls == 0


class _ConstantBearingConvBatchNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        marker = 1
        return self.bn(self.conv(inputs))


def test_rejects_replaced_transitive_dis_helper_before_it_runs() -> None:
    model = _ConstantBearingConvBatchNorm().eval()
    original = dis._get_const_value
    helper_calls = 0

    def hostile_get_const_value(*args: object, **kwargs: object) -> object:
        nonlocal helper_calls
        helper_calls += 1
        return original(*args, **kwargs)

    dis._get_const_value = hostile_get_const_value
    caught: UnsupportedOpError | None = None
    try:
        try:
            parse_model(model, input_shape=(1, 2, 2))
        except UnsupportedOpError as exc:
            caught = exc
    finally:
        dis._get_const_value = original

    assert caught is not None
    assert "dis" in str(caught).lower()
    assert helper_calls == 0


@pytest.mark.parametrize(
    "state_name",
    ["_inline_cache_entries", "hasconst", "sys"],
)
def test_rejects_modified_dis_runtime_state_before_custom_code_runs(
    state_name: str,
) -> None:
    model = _ConstantBearingConvBatchNorm().eval()
    called = 0

    if state_name == "_inline_cache_entries":
        original = dis._inline_cache_entries

        class TrackingList(list[int]):
            def __getitem__(self, index: object) -> object:
                nonlocal called
                called += 1
                raise AssertionError("modified dis table must not be read")

        dis._inline_cache_entries = TrackingList(original)

        def restore() -> None:
            dis._inline_cache_entries = original

    elif state_name == "hasconst":
        original_entry = dis.hasconst[0]

        class TrackingInt(int):
            def __eq__(self, other: object) -> bool:
                nonlocal called
                called += 1
                raise AssertionError("modified dis entry must not be compared")

        dis.hasconst[0] = TrackingInt(original_entry)

        def restore() -> None:
            dis.hasconst[0] = original_entry

    else:
        original_module = dis.sys

        class TrackingSys:
            def __getattr__(self, name: str) -> object:
                nonlocal called
                called += 1
                raise AssertionError(
                    f"modified dis sys binding must not be read: {name}"
                )

        dis.sys = TrackingSys()

        def restore() -> None:
            dis.sys = original_module

    caught: UnsupportedOpError | None = None
    try:
        try:
            parse_model(model, input_shape=(1, 2, 2))
        except UnsupportedOpError as exc:
            caught = exc
    finally:
        restore()

    assert caught is not None
    assert "dis" in str(caught).lower()
    assert called == 0


@pytest.mark.parametrize("class_name", ["Bytecode", "Instruction"])
def test_rejects_modified_dis_class_before_custom_constructor_runs(
    class_name: str,
) -> None:
    model = _ConstantBearingConvBatchNorm().eval()
    cls = dis.Bytecode if class_name == "Bytecode" else dis._Instruction
    attribute_name = "__init__" if class_name == "Bytecode" else "__new__"
    namespace = type.__getattribute__(cls, "__dict__")
    original = namespace[attribute_name]
    called = 0

    def tracking_new(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called += 1
        raise AssertionError("modified dis constructor must not execute")

    replacement: object = tracking_new
    if attribute_name == "__new__":
        replacement = staticmethod(tracking_new)
    setattr(cls, attribute_name, replacement)
    caught: UnsupportedOpError | None = None
    try:
        try:
            parse_model(model, input_shape=(1, 2, 2))
        except UnsupportedOpError as exc:
            caught = exc
    finally:
        setattr(cls, attribute_name, original)

    assert caught is not None
    assert "dis" in str(caught).lower()
    assert called == 0


@pytest.mark.parametrize("tensor_type", [torch.Tensor, nn.Parameter])
def test_rejects_replaced_tensor_copy_binding_before_it_runs(
    tensor_type: type[torch.Tensor],
) -> None:
    model = _basic_model()
    original = type.__getattribute__(tensor_type, "__dict__")["__deepcopy__"]
    called = 0

    def tracking_deepcopy(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called += 1
        raise AssertionError("modified tensor copy binding must not execute")

    setattr(tensor_type, "__deepcopy__", tracking_deepcopy)
    caught: UnsupportedOpError | None = None
    try:
        try:
            parse_model(model, input_shape=(1, 2, 2))
        except UnsupportedOpError as exc:
            caught = exc
    finally:
        setattr(tensor_type, "__deepcopy__", original)

    assert caught is not None
    assert "copy" in str(caught).lower()
    assert called == 0


@pytest.mark.parametrize(
    "dis_class",
    [dis.Positions, dis._Instruction, dis._ExceptionTableEntry],
    ids=["positions", "instruction", "exception-entry"],
)
def test_rejects_modified_dis_constructor_global_before_it_runs(
    dis_class: type[tuple[object, ...]],
) -> None:
    model = _ConstantBearingConvBatchNorm().eval()
    descriptor = type.__getattribute__(dis_class, "__dict__")["__new__"]
    constructor = descriptor.__func__
    globals_state = constructor.__globals__
    original = globals_state["_tuple_new"]
    called = 0

    def tracking_tuple_new(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called += 1
        raise AssertionError("modified dis constructor global must not execute")

    globals_state["_tuple_new"] = tracking_tuple_new
    caught: UnsupportedOpError | None = None
    try:
        try:
            parse_model(model, input_shape=(1, 2, 2))
        except UnsupportedOpError as exc:
            caught = exc
    finally:
        globals_state["_tuple_new"] = original

    assert caught is not None
    assert "dis" in str(caught).lower()
    assert called == 0


@pytest.mark.parametrize(
    ("tensor_type", "global_name"),
    [
        pytest.param(nn.Parameter, "torch", id="parameter-torch-module"),
        pytest.param(
            torch.Tensor,
            "has_torch_function_unary",
            id="tensor-dispatch-helper",
        ),
    ],
)
def test_rejects_modified_tensor_copy_global_before_it_runs(
    tensor_type: type[torch.Tensor],
    global_name: str,
) -> None:
    model = _basic_model()
    function = type.__getattribute__(tensor_type, "__dict__")["__deepcopy__"]
    globals_state = function.__globals__
    original = globals_state[global_name]
    called = 0

    def tracking_global(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called += 1
        raise AssertionError("modified tensor copy global must not execute")

    globals_state[global_name] = tracking_global
    caught: UnsupportedOpError | None = None
    try:
        try:
            parse_model(model, input_shape=(1, 2, 2))
        except UnsupportedOpError as exc:
            caught = exc
    finally:
        globals_state[global_name] = original

    assert caught is not None
    assert "copy" in str(caught).lower()
    assert called == 0


@pytest.mark.parametrize("module", [dis, nn], ids=["dis", "torch-nn"])
def test_rejects_modified_runtime_module_type_without_attribute_access(
    module: ModuleType,
) -> None:
    model = _basic_model()
    original_type = type(module)
    called = 0

    class TrackingModule(ModuleType):
        def __getattribute__(self, name: str) -> object:
            nonlocal called
            called += 1
            raise AssertionError(
                f"modified runtime module attribute must not be read: {name}"
            )

    module.__class__ = TrackingModule
    caught: UnsupportedOpError | None = None
    try:
        try:
            parse_model(model, input_shape=(1, 2, 2))
        except UnsupportedOpError as exc:
            caught = exc
    finally:
        ModuleType.__setattr__(module, "__class__", original_type)

    assert caught is not None
    assert "module type" in str(caught).lower()
    assert called == 0
