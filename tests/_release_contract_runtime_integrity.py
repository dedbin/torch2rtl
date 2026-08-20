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



def test_frontend_rejects_non_cpu_parameter_device_before_lowering() -> None:
    model = nn.Sequential(nn.Linear(2, 2, device="meta"))

    with pytest.raises(UnsupportedOpError, match="device"):
        parse_model(model, input_shape=(2,))


def test_frontend_rejects_noncontiguous_parameter_before_lowering() -> None:
    layer = nn.Linear(3, 2, bias=False)
    base = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    layer.weight = nn.Parameter(base.T)
    assert not layer.weight.is_contiguous()

    with pytest.raises(UnsupportedOpError, match="contiguous"):
        parse_model(nn.Sequential(layer), input_shape=(3,))


@pytest.mark.parametrize(
    ("module", "attribute", "input_shape"),
    [
        (nn.Linear(2, 2), "weight", (2,)),
        (nn.Linear(2, 2), "bias", (2,)),
        (nn.Conv2d(1, 1, kernel_size=1), "weight", (1, 2, 2)),
        (nn.Conv2d(1, 1, kernel_size=1), "bias", (1, 2, 2)),
    ],
)
def test_frontend_rejects_parameter_negative_view_bit(
    module: nn.Module,
    attribute: str,
    input_shape: tuple[int, ...],
) -> None:
    original = getattr(module, attribute)
    assert isinstance(original, nn.Parameter)
    setattr(module, attribute, nn.Parameter(torch._neg_view(torch.ones_like(original))))

    with pytest.raises(UnsupportedOpError, match="negative view"):
        parse_model(nn.Sequential(module), input_shape=input_shape)


def test_frontend_rejects_parameter_gradient_state() -> None:
    layer = nn.Linear(2, 2)
    layer.weight.grad = torch._neg_view(torch.ones_like(layer.weight))

    with pytest.raises(UnsupportedOpError, match="gradient"):
        parse_model(nn.Sequential(layer), input_shape=(2,))


def test_frontend_rejects_ndarray_subclass_instance_state() -> None:
    class StatefulArray(np.ndarray):
        def __bool__(self) -> bool:
            global _ARRAY_STATE_TRACE_SWITCH
            _ARRAY_STATE_TRACE_SWITCH += 1
            return _ARRAY_STATE_TRACE_SWITCH == 1

    class ArrayState(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.flag = np.asarray([1]).view(StatefulArray)
            self.first = nn.Linear(2, 2)
            self.second = nn.Linear(2, 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            layer = self.first if self.flag else self.second
            return layer(inputs)

    global _ARRAY_STATE_TRACE_SWITCH
    _ARRAY_STATE_TRACE_SWITCH = 0
    try:
        with pytest.raises(UnsupportedOpError, match="external Python state"):
            parse_model(ArrayState().eval(), input_shape=(2,))
        assert _ARRAY_STATE_TRACE_SWITCH == 0
    finally:
        _ARRAY_STATE_TRACE_SWITCH = 0


def test_frontend_rejects_primitive_subclass_instance_state() -> None:
    class StatefulInt(int):
        def __bool__(self) -> bool:
            global _INT_STATE_TRACE_SWITCH
            _INT_STATE_TRACE_SWITCH += 1
            return _INT_STATE_TRACE_SWITCH == 1

    class IntegerState(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.flag = StatefulInt(1)
            self.first = nn.Linear(2, 2)
            self.second = nn.Linear(2, 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            layer = self.first if self.flag else self.second
            return layer(inputs)

    global _INT_STATE_TRACE_SWITCH
    _INT_STATE_TRACE_SWITCH = 0
    try:
        with pytest.raises(UnsupportedOpError, match="external Python state"):
            parse_model(IntegerState().eval(), input_shape=(2,))
        assert _INT_STATE_TRACE_SWITCH == 0
    finally:
        _INT_STATE_TRACE_SWITCH = 0


@pytest.mark.parametrize("kind", ["ndarray", "tensor"])
def test_frontend_rejects_direct_tensor_like_instance_state(kind: str) -> None:
    class MetadataState(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            value: object = np.asarray([1], dtype=np.int64)
            if kind == "tensor":
                value = torch.tensor([1.0], requires_grad=True)
            object.__setattr__(self, "flag", value)
            self.first = nn.Linear(2, 2)
            self.second = nn.Linear(2, 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            if kind == "ndarray":
                self.flag.flags.writeable = not self.flag.flags.writeable
                enabled = self.flag.flags.writeable
            else:
                self.flag.requires_grad_(not self.flag.requires_grad)
                enabled = self.flag.requires_grad
            return self.first(inputs) if enabled else self.second(inputs)

    with pytest.raises(UnsupportedOpError, match="external Python state"):
        parse_model(MetadataState().eval(), input_shape=(2,))


@pytest.mark.parametrize("registry", ["state", "modules", "parameters"])
def test_frontend_rejects_non_exact_module_state_registry_names(
    registry: str,
) -> None:
    class BadName(str):
        def __format__(self, _spec: str) -> str:
            raise ValueError("name format trap")

    model = nn.Sequential(nn.Linear(2, 2))
    if registry == "state":
        object.__getattribute__(model, "__dict__")[1] = 0
    elif registry == "modules":
        modules = object.__getattribute__(model, "__dict__")["_modules"]
        modules[BadName("0")] = modules.pop("0")
    else:
        layer = model[0]
        parameters = object.__getattribute__(layer, "__dict__")["_parameters"]
        parameters[BadName("weight")] = parameters.pop("weight")

    with pytest.raises(UnsupportedOpError, match="non-string"):
        parse_model(model, input_shape=(2,))


def test_frontend_rejects_custom_module_metaclass() -> None:
    class SpoofMeta(type):
        def __getattribute__(cls, name: str) -> object:
            if name == "__mro__":
                return (nn.Module, object)
            return super().__getattribute__(name)

    class SpoofedModule(nn.Module, metaclass=SpoofMeta):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.linear(inputs)

    with pytest.raises(UnsupportedOpError, match="metaclass"):
        parse_model(SpoofedModule().eval(), input_shape=(2,))


def test_frontend_rejects_internal_tensor_registry_access_before_trace() -> None:
    class RegistryState(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(2, 2)
            self.second = nn.Linear(2, 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            flag = self.first._parameters["weight"]
            if flag is None:
                return inputs
            flag.requires_grad_(not flag.requires_grad)
            return self.first(inputs) if flag.requires_grad else self.second(inputs)

    with pytest.raises(UnsupportedOpError, match="internal module registry"):
        parse_model(RegistryState().eval(), input_shape=(2,))


def test_frontend_rejects_monkeypatched_framework_forward() -> None:
    original_forward = nn.Linear.forward

    def hidden_forward(
        self: nn.Linear,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        result = original_forward(self, inputs)
        if bool(torch.sum(inputs) != 0):
            return result + 11.0
        return result

    nn.Linear.forward = hidden_forward
    try:
        with pytest.raises(UnsupportedOpError, match="framework"):
            parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    finally:
        nn.Linear.forward = original_forward


def test_frontend_rejects_in_place_framework_function_code_patch() -> None:
    original_code = nn.Linear.forward.__code__

    def replacement(self: nn.Linear, inputs: torch.Tensor) -> torch.Tensor:
        del self
        return inputs

    nn.Linear.forward.__code__ = replacement.__code__
    try:
        with pytest.raises(UnsupportedOpError, match="framework"):
            parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    finally:
        nn.Linear.forward.__code__ = original_code


def test_frontend_rejects_framework_function_globals_patch() -> None:
    framework_globals = nn.Linear.forward.__globals__
    original_functional = framework_globals["F"]

    class FunctionalWrapper:
        linear = staticmethod(torch.nn.functional.linear)

    framework_globals["F"] = FunctionalWrapper
    try:
        with pytest.raises(UnsupportedOpError, match="framework"):
            parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    finally:
        framework_globals["F"] = original_functional


def test_frontend_rejects_monkeypatched_functional_kernel() -> None:
    functional = torch.nn.functional
    original_linear = functional.linear

    def replacement(
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return original_linear(inputs, weight, bias) + 1.0

    functional.linear = replacement
    try:
        with pytest.raises(UnsupportedOpError, match="framework"):
            parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    finally:
        functional.linear = original_linear


def test_frontend_rejects_monkeypatched_torch_relu_kernel() -> None:
    original_relu = torch.relu

    def replacement(inputs: torch.Tensor) -> torch.Tensor:
        return original_relu(inputs) + 1.0

    torch.relu = replacement
    try:
        with pytest.raises(UnsupportedOpError, match="framework"):
            parse_model(nn.Sequential(nn.ReLU()), input_shape=(2,))
    finally:
        torch.relu = original_relu


@pytest.mark.parametrize("attribute", ["numpy", "detach"])
def test_frontend_rejects_monkeypatched_tensor_conversion_method(
    attribute: str,
) -> None:
    had_own_attribute = attribute in vars(torch.Tensor)
    original = getattr(torch.Tensor, attribute)

    if attribute == "numpy":
        def replacement(tensor: torch.Tensor, *args: object, **kwargs: object) -> object:
            return np.zeros_like(original(tensor, *args, **kwargs))
    else:
        def replacement(tensor: torch.Tensor, *args: object, **kwargs: object) -> object:
            return torch.zeros_like(original(tensor, *args, **kwargs))

    setattr(torch.Tensor, attribute, replacement)
    try:
        with pytest.raises(UnsupportedOpError, match="framework"):
            parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    finally:
        if had_own_attribute:
            setattr(torch.Tensor, attribute, original)
        else:
            delattr(torch.Tensor, attribute)


def test_frontend_rejects_monkeypatched_tensor_metadata_property() -> None:
    attribute = "device"
    had_own_attribute = attribute in vars(torch.Tensor)
    original = inspect.getattr_static(torch.Tensor, attribute)
    setattr(torch.Tensor, attribute, property(lambda _tensor: torch.device("cpu")))
    try:
        model = nn.Sequential(nn.Linear(2, 2, device="meta"))
        with pytest.raises(UnsupportedOpError, match="framework"):
            parse_model(model, input_shape=(2,))
    finally:
        if had_own_attribute:
            setattr(torch.Tensor, attribute, original)
        else:
            delattr(torch.Tensor, attribute)


def test_frontend_rejects_monkeypatched_torch_dtype_constants() -> None:
    original_float32 = torch.float32
    original_float64 = torch.float64
    torch.float32 = original_float64
    torch.float64 = original_float32
    try:
        with pytest.raises(UnsupportedOpError, match="framework"):
            parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    finally:
        torch.float32 = original_float32
        torch.float64 = original_float64


def test_frontend_rejects_monkeypatched_default_dtype_binding() -> None:
    original = torch.get_default_dtype
    torch.get_default_dtype = lambda: torch.float32
    try:
        with pytest.raises(UnsupportedOpError, match="framework"):
            parse_model(nn.Sequential(nn.ReLU()), input_shape=(2,))
    finally:
        torch.get_default_dtype = original


def test_frontend_rejects_unhashable_or_hostile_input_dtype_cleanly() -> None:
    class HostileDtype:
        def __hash__(self) -> int:
            raise ValueError("dtype hash trap")

        def __repr__(self) -> str:
            raise ValueError("dtype repr trap")

    with pytest.raises(UnsupportedOpError, match="input dtype"):
        parse_model(nn.Sequential(nn.ReLU()), input_shape=(2,), input_dtype=HostileDtype())


def test_frontend_rejects_global_module_registration_hook() -> None:
    handle = torch.nn.modules.module.register_module_module_registration_hook(
        lambda _parent, _name, module: module
    )
    try:
        with pytest.raises(UnsupportedOpError, match="global.*hook"):
            parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    finally:
        handle.remove()


def test_frontend_rejects_monkeypatched_symbolic_trace() -> None:
    original_trace = torch.fx.symbolic_trace
    torch.fx.symbolic_trace = lambda model: original_trace(model)
    try:
        with pytest.raises(UnsupportedOpError, match="framework"):
            parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    finally:
        torch.fx.symbolic_trace = original_trace


def test_frontend_rejects_replaced_torch_fx_owner_module() -> None:
    original_fx = torch.fx
    model = nn.Sequential(nn.Linear(2, 2, bias=False))
    fake_model = nn.Sequential(nn.Linear(2, 2, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0, 1.0], [2.0, 2.0]]))
        fake_model[0].weight.zero_()

    def fake_trace(_model: nn.Module) -> torch.fx.GraphModule:
        return original_fx.symbolic_trace(fake_model)

    torch.fx = types.SimpleNamespace(symbolic_trace=fake_trace)
    try:
        with pytest.raises(UnsupportedOpError, match="framework"):
            parse_model(model, input_shape=(2,))
    finally:
        torch.fx = original_fx


@pytest.mark.parametrize("owner", ["nn", "nn.modules"])
def test_frontend_rejects_replaced_torch_parent_modules(owner: str) -> None:
    model = nn.Sequential(nn.Linear(2, 2))
    if owner == "nn":
        original = torch.nn
        torch.nn = types.SimpleNamespace()
    else:
        original = torch.nn.modules
        torch.nn.modules = types.SimpleNamespace()
    try:
        with pytest.raises(UnsupportedOpError, match="framework"):
            parse_model(model, input_shape=(2,))
    finally:
        if owner == "nn":
            torch.nn = original
        else:
            torch.nn.modules = original


def test_frontend_rejects_monkeypatched_fx_tracer_method() -> None:
    original_trace = torch.fx.Tracer.trace

    def replacement(
        self: torch.fx.Tracer,
        root: torch.nn.Module,
        concrete_args: dict[str, object] | None = None,
    ) -> torch.fx.Graph:
        return original_trace(self, root, concrete_args)

    torch.fx.Tracer.trace = replacement
    try:
        with pytest.raises(UnsupportedOpError, match="framework"):
            parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    finally:
        torch.fx.Tracer.trace = original_trace


def test_frontend_wraps_non_traceerror_fx_failures() -> None:
    class ProxyLength(nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs if len(inputs) else inputs

    with pytest.raises(UnsupportedOpError, match="FX symbolic trace"):
        parse_model(ProxyLength(), input_shape=(2,))


def test_frontend_rejects_proxy_introspection_with_exception_control_flow() -> None:
    class ProxyIntrospection(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(2, 2, bias=False)
            self.second = nn.Linear(2, 2, bias=False)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            try:
                inputs.node
            except:
                return self.second(inputs)
            return self.first(inputs)

    with pytest.raises(UnsupportedOpError, match="exception handling|introspection"):
        parse_model(ProxyIntrospection().eval(), input_shape=(2,))


def test_frontend_rejects_proxy_dependent_string_formatting() -> None:
    class ProxyFormatting(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(2, 2, bias=False)
            self.second = nn.Linear(2, 2, bias=False)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            marker = f"{inputs}"
            return self.first(inputs) if "Proxy" in marker else self.second(inputs)

    with pytest.raises(UnsupportedOpError, match="formatting"):
        parse_model(ProxyFormatting().eval(), input_shape=(2,))


def test_frontend_rejects_custom_conditional_control_flow() -> None:
    class ProbeAwareBranch(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(2, 2, bias=False)
            self.second = nn.Linear(2, 2, bias=False)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            marker = "%s" % inputs
            if "Proxy" in marker or inputs.sum() == 0:
                return self.first(inputs)
            return self.second(inputs)

    with pytest.raises(UnsupportedOpError, match="control flow"):
        parse_model(ProbeAwareBranch().eval(), input_shape=(2,))


def test_frontend_rejects_nested_dynamic_code_objects() -> None:
    class NestedCode(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(2, 2, bias=False)
            self.second = nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.first.weight.zero_()
                self.second.weight.copy_(torch.tensor([[1.0, 1.0], [2.0, 2.0]]))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return (
                lambda: self.first(inputs)
                if _bump_nested_code_trace_switch() == 1
                else self.second(inputs)
            )()

    global _NESTED_CODE_TRACE_SWITCH
    _NESTED_CODE_TRACE_SWITCH = 0
    try:
        with pytest.raises(UnsupportedOpError, match="nested code"):
            parse_model(NestedCode().eval(), input_shape=(2,))
    finally:
        _NESTED_CODE_TRACE_SWITCH = 0


def test_frontend_rejects_custom_builtins_mapping() -> None:
    class Hidden(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(2, 2, bias=False)

    def template(self: Hidden, inputs: torch.Tensor) -> torch.Tensor:
        len(inputs)
        return self.first(inputs)

    def hidden_len(_value: object) -> int:
        return 1

    Hidden.forward = types.FunctionType(
        template.__code__,
        {"__builtins__": {"len": hidden_len}},
        name="forward",
    )

    with pytest.raises(UnsupportedOpError, match="builtins"):
        parse_model(Hidden().eval(), input_shape=(2,))


def test_frontend_rejects_callable_hidden_in_code_constants() -> None:
    class Hidden(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(2, 2, bias=False)

    def template(self: Hidden, inputs: torch.Tensor) -> torch.Tensor:
        marker = (None,)
        return self.first(inputs)

    class Trigger:
        def __call__(self) -> None:
            return None

    constants = tuple(
        (Trigger(),) if constant == (None,) else constant
        for constant in template.__code__.co_consts
    )
    Hidden.forward = types.FunctionType(
        template.__code__.replace(co_consts=constants),
        template.__globals__,
        name="forward",
    )

    with pytest.raises(UnsupportedOpError, match="constant"):
        parse_model(Hidden().eval(), input_shape=(2,))


def test_frontend_rejects_branchless_proxy_dependent_selection() -> None:
    class BranchlessSelection(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(2, 2, bias=False)
            self.second = nn.Linear(2, 2, bias=False)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            marker = "%s" % inputs
            return (self.first, self.second)["Proxy" not in marker](inputs)

    with pytest.raises(UnsupportedOpError, match="bytecode|string formatting"):
        parse_model(BranchlessSelection().eval(), input_shape=(2,))
