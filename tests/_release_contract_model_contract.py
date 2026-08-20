from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import subprocess
import sys
import tomllib
import types
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

import torch2rtl.cli as cli
import torch2rtl.frontend.pytorch_fx as fx_frontend
from torch2rtl.backend.systemverilog.emit import emit_systemverilog
from torch2rtl.frontend.pytorch_fx import UnsupportedOpError, parse_model
from torch2rtl.ir.ops import ArgmaxIR, Conv2dIR, ReluIR
from torch2rtl.quant.fixed_point import FixedPointConfig, quantize_array
from torch2rtl.quant.reference import (
    QuantizedConv2dIR,
    QuantizedGraph,
    QuantizedLinearIR,
    conv2d_fixed,
    infer_float_graph,
    infer_quantized,
    linear_fixed,
    quantize_graph,
    validate_accumulator_width,
)
from torch2rtl.synth.report import update_report
from torch2rtl.synth.yosys import SynthResult, run_yosys
from torch2rtl.verify.simulator import SimulationResult, _simulation_ok, run_simulation


_TRACE_GLOBAL_SWITCH = 0
_HELPER_TRACE_SWITCH = 0
_DESCRIPTOR_TRACE_SWITCH = 0
_SPOOFED_MODULE_TRACE_SWITCH = 0
_ARRAY_STATE_TRACE_SWITCH = 0
_INT_STATE_TRACE_SWITCH = 0
_STATIC_DESCRIPTOR_TRACE_SWITCH = 0
_NESTED_CODE_TRACE_SWITCH = 0


def _bump_nested_code_trace_switch() -> int:
    global _NESTED_CODE_TRACE_SWITCH
    _NESTED_CODE_TRACE_SWITCH += 1
    return _NESTED_CODE_TRACE_SWITCH



@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.complex64])
def test_frontend_rejects_unsupported_model_float_dtype(dtype: torch.dtype) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = nn.Sequential(nn.Linear(2, 2)).to(dtype=dtype)

    with pytest.raises(UnsupportedOpError, match="dtype"):
        parse_model(model, input_shape=(2,))


@pytest.mark.parametrize("hook_kind", ["forward", "forward_pre"])
@pytest.mark.parametrize("on_root", [False, True], ids=["child", "root"])
def test_frontend_rejects_unlowered_forward_hooks(
    hook_kind: str,
    on_root: bool,
) -> None:
    model = nn.Sequential(nn.Linear(2, 2)).eval()
    target = model if on_root else model[0]
    if hook_kind == "forward":
        target.register_forward_hook(lambda _module, _args, output: output.flip(-1))
    else:
        target.register_forward_pre_hook(
            lambda _module, args: (args[0].flip(-1),)
        )

    with pytest.raises(UnsupportedOpError, match="hook"):
        parse_model(model, input_shape=(2,))


def test_frontend_rejects_instance_forward_override() -> None:
    model = nn.Sequential(nn.Linear(2, 2)).eval()
    layer = model[0]
    original = layer.forward
    layer.forward = types.MethodType(
        lambda _self, inputs: original(inputs).flip(-1),
        layer,
    )

    with pytest.raises(UnsupportedOpError, match="forward"):
        parse_model(model, input_shape=(2,))


def test_frontend_rejects_root_instance_forward_override() -> None:
    class Root(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.linear(inputs)

    model = Root().eval()
    original = model.forward
    model.forward = types.MethodType(
        lambda _self, inputs: original(inputs).flip(-1),
        model,
    )

    with pytest.raises(UnsupportedOpError, match="forward"):
        parse_model(model, input_shape=(2,))


@pytest.mark.parametrize("override", ["__call__", "_call_impl"])
def test_frontend_rejects_custom_root_call_path(override: str) -> None:
    class Root(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.linear(inputs)

    if override == "__call__":
        class CustomCall(Root):
            def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
                return super().__call__(inputs).flip(-1)

        model: nn.Module = CustomCall()
    else:
        class CustomCallImpl(Root):
            def _call_impl(self, *args: object, **kwargs: object) -> torch.Tensor:
                return super()._call_impl(*args, **kwargs).flip(-1)

        model = CustomCallImpl()

    with pytest.raises(UnsupportedOpError, match="call"):
        parse_model(model.eval(), input_shape=(2,))


def test_frontend_rejects_inplace_relu_side_effect() -> None:
    with pytest.raises(UnsupportedOpError, match="inplace"):
        parse_model(nn.Sequential(nn.ReLU(inplace=True)), input_shape=(2, 3))


def test_frontend_rejects_python_instance_state_mutated_during_trace() -> None:
    class Stateful(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.counter = 0
            self.first = nn.Linear(2, 2)
            self.second = nn.Linear(2, 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            self.counter += 1
            layer = self.first if self.counter == 1 else self.second
            return layer(inputs)

    model = Stateful().eval()

    with pytest.raises(UnsupportedOpError, match="state"):
        parse_model(model, input_shape=(2,))

    assert model.counter == 0


def test_frontend_rejects_custom_class_level_state_before_trace() -> None:
    class ClassState(nn.Module):
        counter = 0

        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            type(self).counter += 1
            return self.linear(inputs)

    with pytest.raises(UnsupportedOpError, match="class-level state"):
        parse_model(ClassState().eval(), input_shape=(2,))

    assert ClassState.counter == 0


def test_frontend_rejects_custom_root_getattr() -> None:
    class CustomGetattr(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 2)

        def __getattr__(self, name: str) -> object:
            if name == "chosen":
                return super().__getattr__("linear")
            return super().__getattr__(name)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.chosen(inputs)  # type: ignore[operator]

    with pytest.raises(UnsupportedOpError, match="__getattr__"):
        parse_model(CustomGetattr().eval(), input_shape=(2,))


def test_frontend_rejects_compiled_call_descriptor_without_executing_it() -> None:
    class CompiledCallDescriptor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 2)

        @property
        def _compiled_call_impl(self) -> object:
            raise AssertionError("validation must not execute the descriptor")

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.linear(inputs)

    with pytest.raises(UnsupportedOpError, match="compiled.*call"):
        parse_model(CompiledCallDescriptor().eval(), input_shape=(2,))


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool])
def test_frontend_rejects_non_floating_parameter_dtype(dtype: torch.dtype) -> None:
    layer = nn.Linear(2, 2, bias=False)
    layer.weight = nn.Parameter(
        torch.ones((2, 2), dtype=dtype),
        requires_grad=False,
    )

    with pytest.raises(UnsupportedOpError, match="dtype"):
        parse_model(nn.Sequential(layer), input_shape=(2,))


def test_frontend_wraps_fx_dynamic_control_flow_error() -> None:
    class DynamicControl(nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs if inputs.sum() > 0 else -inputs

    with pytest.raises(UnsupportedOpError, match="FX symbolic trace"):
        parse_model(DynamicControl(), input_shape=(2,))


def test_frontend_rejects_instance_call_impl_on_leaf_module() -> None:
    model = nn.Sequential(nn.Linear(2, 2)).eval()
    layer = model[0]
    original = layer._call_impl
    layer._call_impl = types.MethodType(
        lambda _self, *args, **kwargs: original(*args, **kwargs).flip(-1),
        layer,
    )

    with pytest.raises(UnsupportedOpError, match="call"):
        parse_model(model, input_shape=(2,))


def test_frontend_rejects_custom_deepcopy_before_it_can_mutate_or_substitute() -> None:
    class CustomDeepcopy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.copy_count = 0
            self.linear = nn.Linear(2, 2)

        def __deepcopy__(self, _memo: dict[int, object]) -> nn.Module:
            self.copy_count += 1
            return nn.Sequential(nn.Linear(2, 2))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.linear(inputs)

    model = CustomDeepcopy().eval()

    with pytest.raises(UnsupportedOpError, match="copy"):
        parse_model(model, input_shape=(2,))

    assert model.copy_count == 0


def test_frontend_rejects_monkeypatched_deepcopy_binding() -> None:
    class Original(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(2, 2, bias=False)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.first(inputs)

    class Substitute(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(2, 2, bias=False)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.first(inputs)

    model = Original().eval()
    substitute = Substitute().eval()
    Substitute.__module__ = Original.__module__
    Substitute.__qualname__ = Original.__qualname__
    original_deepcopy = fx_frontend.copy.deepcopy
    fx_frontend.copy.deepcopy = lambda _value: substitute
    try:
        with pytest.raises(UnsupportedOpError, match="copy"):
            parse_model(model, input_shape=(2,))
    finally:
        fx_frontend.copy.deepcopy = original_deepcopy


def test_frontend_rejects_nested_module_class_state_before_trace() -> None:
    class Inner(nn.Module):
        counter = 0

        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            type(self).counter += 1
            return self.linear(inputs)

    class Outer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = Inner()

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.inner(inputs)

    with pytest.raises(UnsupportedOpError, match="class-level state"):
        parse_model(Outer().eval(), input_shape=(2,))

    assert Inner.counter == 0


def test_frontend_does_not_trust_overridden_parameter_introspection() -> None:
    class HiddenParameters(nn.Sequential):
        def named_parameters(self, *args: object, **kwargs: object) -> object:
            return iter(())

    layer = nn.Linear(2, 2, bias=False).to(dtype=torch.float16)

    with pytest.raises(UnsupportedOpError, match="dtype"):
        parse_model(HiddenParameters(layer), input_shape=(2,))


def test_frontend_rejects_parameter_subclass_torch_function_semantics() -> None:
    class HiddenLinearEffect(nn.Parameter):
        @classmethod
        def __torch_function__(
            cls,
            func: object,
            types: tuple[type[object], ...],
            args: tuple[object, ...] = (),
            kwargs: dict[str, object] | None = None,
        ) -> object:
            call_kwargs = {} if kwargs is None else kwargs
            result = super().__torch_function__(func, types, args, call_kwargs)
            if getattr(func, "__name__", "") == "linear":
                input_tensor = args[0]
                if bool(torch.sum(input_tensor) != 0):
                    return result + 7.0
            return result

    model = nn.Sequential(nn.Linear(2, 2, bias=False)).eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(2))
    model[0].weight = HiddenLinearEffect(
        model[0].weight.detach(),
        requires_grad=False,
    )

    with pytest.raises(UnsupportedOpError, match="subclass"):
        parse_model(model, input_shape=(2,))


def test_frontend_rejects_autocast_decorated_forward() -> None:
    class AutocastModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 2)

        @torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.linear(inputs)

    with pytest.raises(UnsupportedOpError, match="decorated|dtype"):
        parse_model(AutocastModel().eval(), input_shape=(2,))


def test_frontend_rejects_external_python_state_before_trace() -> None:
    class GlobalState(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(2, 2)
            self.second = nn.Linear(2, 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            global _TRACE_GLOBAL_SWITCH
            _TRACE_GLOBAL_SWITCH += 1
            layer = self.first if _TRACE_GLOBAL_SWITCH == 1 else self.second
            return layer(inputs)

    global _TRACE_GLOBAL_SWITCH
    _TRACE_GLOBAL_SWITCH = 0
    try:
        with pytest.raises(UnsupportedOpError, match="external Python state"):
            parse_model(GlobalState().eval(), input_shape=(2,))
        assert _TRACE_GLOBAL_SWITCH == 0
    finally:
        _TRACE_GLOBAL_SWITCH = 0


def test_frontend_rejects_external_state_in_reachable_helper_before_trace() -> None:
    class HelperState(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(2, 2)
            self.second = nn.Linear(2, 2)

        def choose(self, inputs: torch.Tensor) -> torch.Tensor:
            global _HELPER_TRACE_SWITCH
            _HELPER_TRACE_SWITCH += 1
            layer = self.first if _HELPER_TRACE_SWITCH == 1 else self.second
            return layer(inputs)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.choose(inputs)

    global _HELPER_TRACE_SWITCH
    _HELPER_TRACE_SWITCH = 0
    try:
        with pytest.raises(UnsupportedOpError, match="external Python state"):
            parse_model(HelperState().eval(), input_shape=(2,))
        assert _HELPER_TRACE_SWITCH == 0
    finally:
        _HELPER_TRACE_SWITCH = 0


def test_frontend_rejects_callable_class_descriptor_before_trace() -> None:
    class CallableDescriptor:
        def __call__(self) -> None:
            return None

        def __get__(self, instance: object, owner: type[object]) -> bool:
            del instance, owner
            global _DESCRIPTOR_TRACE_SWITCH
            _DESCRIPTOR_TRACE_SWITCH += 1
            return _DESCRIPTOR_TRACE_SWITCH == 1

    class DescriptorState(nn.Module):
        choose_first = CallableDescriptor()

        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(2, 2)
            self.second = nn.Linear(2, 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            layer = self.first if self.choose_first else self.second
            return layer(inputs)

    global _DESCRIPTOR_TRACE_SWITCH
    _DESCRIPTOR_TRACE_SWITCH = 0
    try:
        with pytest.raises(UnsupportedOpError, match="class-level state|descriptor"):
            parse_model(DescriptorState().eval(), input_shape=(2,))
        assert _DESCRIPTOR_TRACE_SWITCH == 0
    finally:
        _DESCRIPTOR_TRACE_SWITCH = 0


def test_frontend_rejects_staticmethod_descriptor_subclass() -> None:
    class StatefulStaticmethod(staticmethod):
        def __get__(self, instance: object, owner: type[object]) -> bool:
            del instance, owner
            global _STATIC_DESCRIPTOR_TRACE_SWITCH
            _STATIC_DESCRIPTOR_TRACE_SWITCH += 1
            return _STATIC_DESCRIPTOR_TRACE_SWITCH == 1

    class StaticDescriptorState(nn.Module):
        choose_first = StatefulStaticmethod(lambda: True)

        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(2, 2)
            self.second = nn.Linear(2, 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            layer = self.first if self.choose_first else self.second
            return layer(inputs)

    global _STATIC_DESCRIPTOR_TRACE_SWITCH
    _STATIC_DESCRIPTOR_TRACE_SWITCH = 0
    try:
        with pytest.raises(UnsupportedOpError, match="descriptor"):
            parse_model(StaticDescriptorState().eval(), input_shape=(2,))
        assert _STATIC_DESCRIPTOR_TRACE_SWITCH == 0
    finally:
        _STATIC_DESCRIPTOR_TRACE_SWITCH = 0


def test_frontend_rejects_mutable_dunder_class_state_before_trace() -> None:
    class DunderState(nn.Module):
        __state__: list[int] = []

        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(2, 2)
            self.second = nn.Linear(2, 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            type(self).__state__.append(1)
            layer = self.first if len(type(self).__state__) == 1 else self.second
            return layer(inputs)

    with pytest.raises(UnsupportedOpError, match="class-level state"):
        parse_model(DunderState().eval(), input_shape=(2,))

    assert DunderState.__state__ == []


def test_frontend_rejects_mutable_forward_default_before_trace() -> None:
    class MutableDefault(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(2, 2)
            self.second = nn.Linear(2, 2)

        def forward(
            self,
            inputs: torch.Tensor,
            counter: list[int] = [],
        ) -> torch.Tensor:
            counter.append(1)
            layer = self.first if len(counter) == 1 else self.second
            return layer(inputs)

    defaults = MutableDefault.forward.__defaults__
    assert defaults is not None
    counter = defaults[0]
    assert isinstance(counter, list) and not counter

    with pytest.raises(UnsupportedOpError, match="external Python state"):
        parse_model(MutableDefault().eval(), input_shape=(2,))

    assert counter == []


def test_frontend_rejects_reachable_function_attribute_state() -> None:
    class FunctionState(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(2, 2)
            self.second = nn.Linear(2, 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            state = self.forward.state  # type: ignore[attr-defined]
            state.append(1)
            layer = self.first if len(state) == 1 else self.second
            return layer(inputs)

    FunctionState.forward.state = []  # type: ignore[attr-defined]

    with pytest.raises(UnsupportedOpError, match="function state"):
        parse_model(FunctionState().eval(), input_shape=(2,))

    assert FunctionState.forward.state == []  # type: ignore[attr-defined]


def test_frontend_rejects_dunder_function_introspection_before_trace() -> None:
    class FunctionAnnotations(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(2, 2)
            self.second = nn.Linear(2, 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            annotations = self.forward.__annotations__
            state = annotations.setdefault("counter", [])
            state.append(1)
            layer = self.first if len(state) == 1 else self.second
            return layer(inputs)

    before = dict(FunctionAnnotations.forward.__annotations__)

    with pytest.raises(UnsupportedOpError, match="introspection"):
        parse_model(FunctionAnnotations().eval(), input_shape=(2,))

    assert FunctionAnnotations.forward.__annotations__ == before


def test_frontend_does_not_trust_spoofed_torch_module_name() -> None:
    class SpoofedModule(nn.Module):
        __module__ = "torch.nn.evil"

        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(2, 2)
            self.second = nn.Linear(2, 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            global _SPOOFED_MODULE_TRACE_SWITCH
            _SPOOFED_MODULE_TRACE_SWITCH += 1
            layer = self.first if _SPOOFED_MODULE_TRACE_SWITCH == 1 else self.second
            return layer(inputs)

    global _SPOOFED_MODULE_TRACE_SWITCH
    _SPOOFED_MODULE_TRACE_SWITCH = 0
    try:
        with pytest.raises(UnsupportedOpError, match="external Python state"):
            parse_model(SpoofedModule().eval(), input_shape=(2,))
        assert _SPOOFED_MODULE_TRACE_SWITCH == 0
    finally:
        _SPOOFED_MODULE_TRACE_SWITCH = 0
