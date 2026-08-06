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


@pytest.mark.parametrize("with_relu", [False, True], ids=["conv", "conv-relu"])
def test_terminal_multidimensional_output_is_flattened_only_in_vector_files(
    tmp_path: Path,
    with_relu: bool,
) -> None:
    layers: list[nn.Module] = [nn.Conv2d(1, 1, kernel_size=2)]
    if with_relu:
        layers.append(nn.ReLU())
    model = nn.Sequential(*layers).eval()
    graph = parse_model(model, input_shape=(1, 3, 3))
    inputs = np.arange(9, dtype=np.float32).reshape(1, 3, 3) / 8

    floating = infer_float_graph(graph, inputs)
    qgraph = quantize_graph(graph, FixedPointConfig())
    fixed = infer_quantized(qgraph, quantize_array(inputs, qgraph.cfg))

    assert graph.output.shape == (1, 2, 2)
    assert floating.output.shape == (1, 2, 2)
    assert floating.logits.shape == (1, 2, 2)
    assert fixed.output.shape == (1, 2, 2)
    assert fixed.logits.shape == (1, 2, 2)
    assert isinstance(graph.ops[-1], ReluIR if with_relu else Conv2dIR)

    emit_systemverilog(graph, qgraph.cfg, tmp_path, vector_count=2, seed=9)
    expected_logits = np.loadtxt(tmp_path / "expected_logits.txt", dtype=np.int64)
    assert expected_logits.shape == (2, 4)
    assert run_simulation(tmp_path).ok


def test_explicit_global_argmax_preserves_model_output_and_pre_argmax_logits(
    tmp_path: Path,
) -> None:
    class ConvArgmax(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 2, kernel_size=2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return torch.argmax(self.conv(inputs))

    model = ConvArgmax().eval()
    inputs = torch.tensor(
        [[[0.25, -0.5, 0.75], [1.0, -0.25, 0.5], [-0.75, 0.125, 0.625]]]
    )
    graph = parse_model(model, input_shape=(1, 3, 3))
    floating = infer_float_graph(graph, inputs.numpy())
    qgraph = quantize_graph(graph, FixedPointConfig())
    fixed = infer_quantized(qgraph, quantize_array(inputs.numpy(), qgraph.cfg))

    assert isinstance(graph.ops[-1], ArgmaxIR)
    assert graph.output.shape == ()
    assert floating.logits.shape == (2, 2, 2)
    assert floating.output.shape == ()
    assert int(floating.output) == floating.class_id == int(model(inputs))
    assert floating.activations[-1].name == graph.ops[-1].name
    assert floating.activations[-1].values.shape == ()
    assert fixed.logits.shape == (2, 2, 2)
    assert fixed.output.shape == ()
    assert int(fixed.output) == fixed.class_id
    assert fixed.activations[-1].values.shape == ()

    emit_systemverilog(graph, qgraph.cfg, tmp_path, vector_count=2, seed=10)
    assert np.loadtxt(tmp_path / "expected_logits.txt").shape == (2, 8)
    assert run_simulation(tmp_path).ok


@pytest.mark.parametrize("operation", ["relu", "argmax"])
def test_fixed_reference_preserves_multidimensional_input_shape_until_flatten(
    operation: str,
) -> None:
    class DirectArgmax(nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return torch.argmax(inputs)

    model: nn.Module = nn.Sequential(nn.ReLU())
    if operation == "argmax":
        model = DirectArgmax()
    inputs = np.asarray(
        [[-1.0, 0.25, 0.5], [0.75, -0.5, 0.125]],
        dtype=np.float32,
    )
    graph = parse_model(model.eval(), input_shape=(2, 3))
    qgraph = quantize_graph(graph, FixedPointConfig())
    result = infer_quantized(qgraph, quantize_array(inputs, qgraph.cfg))

    assert result.logits.shape == (2, 3)
    if operation == "relu":
        assert result.output.shape == (2, 3)
        assert result.activations[-1].values.shape == (2, 3)
    else:
        assert graph.output.dtype == "int64"
        assert result.output.shape == ()


def test_float32_reference_does_not_change_class_by_promoting_to_float64() -> None:
    model = nn.Sequential(nn.Linear(3, 2)).eval()
    with torch.no_grad():
        model[0].weight.copy_(
            torch.tensor([[1.0e8, 1.0, -1.0e8], [0.0, 0.0, 0.0]])
        )
        model[0].bias.copy_(torch.tensor([0.0, 0.5]))
    inputs = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    graph = parse_model(model, input_shape=(3,))
    result = infer_float_graph(graph, inputs.numpy())
    expected = model(inputs).detach().numpy()

    assert graph.input.dtype == "float32"
    assert graph.output.dtype == "float32"
    assert graph.ops[0].weight.dtype == np.float32
    assert result.logits.dtype == np.float32
    np.testing.assert_array_equal(result.output, expected)
    assert result.class_id == int(torch.argmax(model(inputs))) == 1


def test_float32_linear_reference_uses_pytorch_accumulation_semantics() -> None:
    weight = torch.tensor(
        [
            1024.0,
            16384.0,
            -0.0625,
            -1024.0,
            16384.0,
            32768.0,
            0.0009765625,
            131072.0,
        ],
        dtype=torch.float32,
    )
    inputs = torch.tensor(
        [
            0.10558542609214783,
            0.373366117477417,
            0.5779691338539124,
            0.5151948928833008,
            -0.011456655338406563,
            -0.39703086018562317,
            0.3128998279571533,
            -0.7188253998756409,
        ],
        dtype=torch.float32,
    )
    model = nn.Sequential(nn.Linear(8, 2)).eval()
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].weight[0].copy_(weight)
        tie_value = torch.nn.functional.linear(inputs, weight.reshape(1, -1)).item()
        model[0].bias.copy_(torch.tensor([0.0, tie_value]))

    graph = parse_model(model, input_shape=(8,))
    result = infer_float_graph(graph, inputs.numpy())
    expected = model(inputs).detach().numpy()

    np.testing.assert_array_equal(result.logits, expected)
    assert result.class_id == int(torch.argmax(model(inputs))) == 0


def test_float32_conv_reference_uses_pytorch_accumulation_semantics() -> None:
    weight = torch.tensor(
        [-2048.0, -65536.0, -512.0, 4194304.0],
        dtype=torch.float32,
    )
    inputs = torch.tensor(
        [
            0.2500721514225006,
            -0.24794362485408783,
            0.5054433941841125,
            0.4355789124965668,
        ],
        dtype=torch.float32,
    ).reshape(1, 1, 4)
    model = nn.Sequential(nn.Conv2d(1, 2, kernel_size=(1, 4))).eval()
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].weight[0, 0, 0].copy_(weight)
        tie_value = torch.nn.functional.conv2d(
            inputs,
            model[0].weight[:1],
            None,
        ).item()
        model[0].bias.copy_(torch.tensor([0.0, tie_value]))

    graph = parse_model(model, input_shape=(1, 1, 4))
    result = infer_float_graph(graph, inputs.numpy())
    expected = model(inputs).detach().numpy()

    np.testing.assert_array_equal(result.logits, expected)
    assert result.class_id == int(torch.argmax(model(inputs))) == 0


def test_float64_model_and_reference_stay_float64() -> None:
    model = nn.Sequential(nn.Linear(3, 2)).double().eval()
    inputs = torch.tensor([0.25, -0.5, 0.75], dtype=torch.float64)
    graph = parse_model(model, input_shape=(3,))
    result = infer_float_graph(graph, inputs.numpy())

    assert graph.input.dtype == "float64"
    assert graph.ops[0].weight.dtype == np.float64
    assert result.logits.dtype == np.float64
    np.testing.assert_allclose(
        result.output,
        model(inputs).detach().numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


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


def test_float_reference_rejects_string_input_instead_of_casting() -> None:
    graph = parse_model(nn.Sequential(nn.ReLU()), input_shape=(2,))

    with pytest.raises(TypeError, match="float32/float64 or integer"):
        infer_float_graph(graph, np.asarray(["1.25", "-0.5"]))


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("kernel_size", (0, 1)),
        ("kernel_size", (-1, 1)),
        ("kernel_size", (True, 1)),
        ("kernel_size", (1.5, 1)),
        ("stride", (0, 1)),
        ("stride", (-1, 1)),
        ("stride", (True, 1)),
        ("padding", (-1, 0)),
        ("padding", (True, 0)),
    ],
)
def test_frontend_rejects_invalid_conv_parameters(
    attribute: str,
    value: object,
) -> None:
    conv = nn.Conv2d(1, 1, kernel_size=1)
    setattr(conv, attribute, value)

    with pytest.raises(UnsupportedOpError, match=attribute):
        parse_model(nn.Sequential(conv), input_shape=(1, 4, 4))


@pytest.mark.parametrize(
    ("attribute", "value"),
    [("start_dim", None), ("start_dim", True), ("end_dim", "-1")],
)
def test_frontend_rejects_non_exact_flatten_dimensions(
    attribute: str,
    value: object,
) -> None:
    flatten = nn.Flatten()
    setattr(flatten, attribute, value)

    with pytest.raises(UnsupportedOpError, match=attribute):
        parse_model(nn.Sequential(flatten), input_shape=(2, 2))


def test_frontend_rejects_input_shape_integer_subclasses() -> None:
    class ShapeDimension(int):
        pass

    with pytest.raises(UnsupportedOpError, match="input shape"):
        parse_model(nn.Sequential(nn.ReLU()), input_shape=(ShapeDimension(2),))


def test_frontend_rejects_hostile_input_shape_without_raw_exception() -> None:
    class HostileDimension:
        def __repr__(self) -> str:
            raise ValueError("shape repr trap")

    class HostileShape:
        def __iter__(self) -> object:
            raise ValueError("shape iteration trap")

    model = nn.Sequential(nn.ReLU())
    with pytest.raises(UnsupportedOpError, match="input shape"):
        parse_model(model, input_shape=(HostileDimension(),))  # type: ignore[arg-type]
    with pytest.raises(UnsupportedOpError, match="input shape"):
        parse_model(model, input_shape=HostileShape())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "input_shape",
    [
        {2},
        iter([2]),
        b"2",
    ],
)
def test_frontend_rejects_non_sequence_shape_containers(input_shape: object) -> None:
    with pytest.raises(UnsupportedOpError, match="input shape"):
        parse_model(  # type: ignore[arg-type]
            nn.Sequential(nn.ReLU()),
            input_shape=input_shape,
        )


@pytest.mark.parametrize("dimension", [2**63, 10**100])
def test_frontend_rejects_unrepresentable_input_dimensions(dimension: int) -> None:
    with pytest.raises(UnsupportedOpError, match="input shape"):
        parse_model(nn.Sequential(nn.ReLU()), input_shape=(dimension,))


def test_frontend_rejects_input_rank_above_numpy_limit() -> None:
    with pytest.raises(UnsupportedOpError, match="input shape"):
        parse_model(nn.Sequential(nn.ReLU()), input_shape=(1,) * 65)


def test_frontend_rejects_conv_structural_int_outside_signed_32_bit() -> None:
    stride = 5_000_000_000
    padding = 2 * stride - 2**32
    conv = nn.Conv2d(1, 1, kernel_size=1)
    conv.stride = (stride, stride)
    conv.padding = (padding, padding)

    with pytest.raises(UnsupportedOpError, match="signed 32-bit"):
        parse_model(nn.Sequential(conv), input_shape=(1, 1, 1))


def test_frontend_rejects_oversized_conv_output_before_probe() -> None:
    conv = nn.Conv2d(1, 1, kernel_size=1, padding=50_000)

    with pytest.raises(UnsupportedOpError, match="element count"):
        parse_model(nn.Sequential(conv), input_shape=(1, 1, 1))


def test_frontend_rejects_conv_attributes_inconsistent_with_parameter_shapes() -> None:
    conv = nn.Conv2d(1, 1, kernel_size=1)
    conv.kernel_size = (2, 2)

    with pytest.raises(UnsupportedOpError, match="weight shape"):
        parse_model(nn.Sequential(conv), input_shape=(1, 4, 4))


@pytest.mark.parametrize(
    ("module", "input_shape"),
    [
        (nn.Linear(2, 2), (2,)),
        (nn.Conv2d(1, 1, kernel_size=1), (1, 2, 2)),
    ],
)
def test_frontend_rejects_missing_required_weight_parameter(
    module: nn.Module,
    input_shape: tuple[int, ...],
) -> None:
    module.register_parameter("weight", None)

    with pytest.raises(UnsupportedOpError, match="weight"):
        parse_model(nn.Sequential(module), input_shape=input_shape)


@pytest.mark.parametrize(
    ("module", "attribute", "input_shape"),
    [
        (nn.Linear(2, 2), "weight", (2,)),
        (nn.Conv2d(1, 1, kernel_size=1), "stride", (1, 2, 2)),
        (nn.ReLU(), "inplace", (2,)),
        (nn.Flatten(), "start_dim", (2, 2)),
    ],
)
def test_frontend_wraps_missing_standard_module_attributes(
    module: nn.Module,
    attribute: str,
    input_shape: tuple[int, ...],
) -> None:
    delattr(module, attribute)

    with pytest.raises(UnsupportedOpError, match=attribute):
        parse_model(nn.Sequential(module), input_shape=input_shape)


@pytest.mark.parametrize(
    ("module", "input_shape", "message"),
    [
        (nn.Linear(2, 0), (2,), "out_features"),
        (nn.Conv2d(1, 0, kernel_size=1), (1, 2, 2), "out_channels"),
    ],
)
def test_frontend_rejects_zero_sized_layer_outputs(
    module: nn.Module,
    input_shape: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(UnsupportedOpError, match=message):
        parse_model(nn.Sequential(module), input_shape=input_shape)


@pytest.mark.parametrize(
    ("module", "attribute", "value", "input_shape"),
    [
        (nn.Linear(2, 2), "in_features", True, (2,)),
        (nn.Linear(2, 2), "out_features", "2", (2,)),
        (nn.Conv2d(1, 1, kernel_size=1), "in_channels", True, (1, 2, 2)),
        (nn.Conv2d(1, 1, kernel_size=1), "out_channels", "1", (1, 2, 2)),
        (nn.Conv2d(1, 1, kernel_size=1), "groups", True, (1, 2, 2)),
    ],
)
def test_frontend_rejects_non_exact_structural_integer_attributes(
    module: nn.Module,
    attribute: str,
    value: object,
    input_shape: tuple[int, ...],
) -> None:
    setattr(module, attribute, value)

    with pytest.raises(UnsupportedOpError, match=attribute):
        parse_model(nn.Sequential(module), input_shape=input_shape)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.asarray([np.iinfo(np.uint64).max], dtype=np.uint64), "range"),
        (np.asarray([True], dtype=np.bool_), "boolean"),
        (np.asarray([1.9], dtype=np.float64), "integer"),
        (np.asarray([np.nan], dtype=np.float64), "finite"),
        (np.asarray([np.inf], dtype=np.float64), "finite"),
        (np.asarray([2**100], dtype=object), "object"),
    ],
)
@pytest.mark.parametrize("position", ["input", "weight", "bias"])
def test_linear_raw_fixed_values_are_validated_before_cast(
    value: np.ndarray,
    message: str,
    position: str,
) -> None:
    inputs: object = np.asarray([1], dtype=np.int64)
    weight: object = np.asarray([[1]], dtype=np.int64)
    bias: object = np.asarray([0], dtype=np.int64)
    if position == "input":
        inputs = value
    elif position == "weight":
        weight = value.reshape(1, 1)
    else:
        bias = value

    with pytest.raises((TypeError, ValueError, OverflowError), match=message):
        linear_fixed(inputs, weight, bias, FixedPointConfig())


@pytest.mark.parametrize("position", ["input", "weight", "bias"])
def test_conv_raw_fixed_values_are_validated_before_cast(position: str) -> None:
    op = QuantizedConv2dIR(
        name="conv",
        in_channels=1,
        out_channels=1,
        input_height=1,
        input_width=1,
        output_height=1,
        output_width=1,
        kernel_height=1,
        kernel_width=1,
        stride=(1, 1),
        padding=(0, 0),
        weight=np.asarray([[[[1]]]], dtype=np.int64),
        bias=np.asarray([0], dtype=np.int64),
    )
    inputs: object = np.asarray([1], dtype=np.int64)
    if position == "input":
        inputs = np.asarray([1.5])
    elif position == "weight":
        object.__setattr__(op, "weight", np.asarray([[[[True]]]]))
    else:
        object.__setattr__(op, "bias", np.asarray([np.inf]))

    with pytest.raises((TypeError, ValueError), match="integer|boolean|finite"):
        conv2d_fixed(inputs, op, FixedPointConfig())


def test_accumulator_analysis_rejects_invalid_raw_parameters_before_cast() -> None:
    qgraph = QuantizedGraph(
        input_shape=(1,),
        cfg=FixedPointConfig(),
        ops=(
            QuantizedLinearIR(
                name="invalid",
                in_features=1,
                out_features=1,
                weight=np.asarray([[np.iinfo(np.uint64).max]], dtype=np.uint64),
                bias=np.asarray([0], dtype=np.int64),
            ),
        ),
    )

    with pytest.raises(OverflowError, match="range"):
        validate_accumulator_width(qgraph)


@pytest.mark.parametrize("bits", [2, 32])
def test_fixed_point_bits_supported_boundaries(bits: int) -> None:
    cfg = FixedPointConfig(bits=bits, frac_bits=bits - 1, acc_bits=64)
    assert cfg.bits == bits


@pytest.mark.parametrize("bits", [1, 33])
def test_fixed_point_bits_rejects_outside_supported_boundaries(bits: int) -> None:
    with pytest.raises(ValueError, match="bits"):
        FixedPointConfig(bits=bits, frac_bits=0, acc_bits=64)


@pytest.mark.parametrize("frac_bits", [0, 7])
def test_fixed_point_frac_bits_supported_boundaries(frac_bits: int) -> None:
    assert FixedPointConfig(bits=8, frac_bits=frac_bits, acc_bits=9).frac_bits == frac_bits


@pytest.mark.parametrize("frac_bits", [-1, 8])
def test_fixed_point_frac_bits_rejects_outside_supported_boundaries(
    frac_bits: int,
) -> None:
    with pytest.raises(ValueError, match="frac_bits"):
        FixedPointConfig(bits=8, frac_bits=frac_bits, acc_bits=9)


@pytest.mark.parametrize("acc_bits", [9, 64])
def test_fixed_point_acc_bits_supported_boundaries(acc_bits: int) -> None:
    assert FixedPointConfig(bits=8, frac_bits=0, acc_bits=acc_bits).acc_bits == acc_bits


@pytest.mark.parametrize("acc_bits", [8, 65])
def test_fixed_point_acc_bits_rejects_outside_supported_boundaries(acc_bits: int) -> None:
    with pytest.raises(ValueError, match="acc_bits"):
        FixedPointConfig(bits=8, frac_bits=0, acc_bits=acc_bits)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bits": True},
        {"bits": 8.0},
        {"frac_bits": True},
        {"frac_bits": 6.0},
        {"acc_bits": True},
        {"acc_bits": 32.0},
    ],
)
def test_fixed_point_config_rejects_non_integer_field_types(
    kwargs: dict[str, object],
) -> None:
    values: dict[str, object] = {"bits": 8, "frac_bits": 6, "acc_bits": 32}
    values.update(kwargs)
    with pytest.raises(TypeError):
        FixedPointConfig(**values)  # type: ignore[arg-type]


def test_quantize_large_finite_float32_saturates_without_warning() -> None:
    cfg = FixedPointConfig(bits=32, frac_bits=31, acc_bits=64)
    values = np.asarray(
        [-np.finfo(np.float32).max, np.finfo(np.float32).max],
        dtype=np.float32,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = quantize_array(values, cfg)

    np.testing.assert_array_equal(result, [cfg.min_int, cfg.max_int])


def test_emit_rejects_zero_vectors_before_writing_artifacts(tmp_path: Path) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))

    with pytest.raises(ValueError, match="vector_count"):
        emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=0)

    assert not tmp_path.exists() or not tuple(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("filename", "mutation"),
    [
        ("input_vectors.txt", "empty"),
        ("expected_classes.txt", "empty"),
        ("expected_logits.txt", "empty"),
        ("input_vectors.txt", "short"),
        ("expected_classes.txt", "short"),
        ("expected_logits.txt", "short"),
        ("input_vectors.txt", "extra"),
        ("expected_classes.txt", "extra"),
        ("expected_logits.txt", "extra"),
    ],
)
def test_simulation_rejects_malformed_vector_file_lengths(
    tmp_path: Path,
    filename: str,
    mutation: str,
) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    path = tmp_path / filename
    tokens = path.read_text(encoding="utf-8").split()
    if mutation == "empty":
        tokens = []
    elif mutation == "short":
        tokens = tokens[:-1]
    else:
        tokens.append("0")
    path.write_text("\n".join(tokens) + ("\n" if tokens else ""), encoding="utf-8")

    result = run_simulation(tmp_path)

    assert not result.ok
    assert result.status == "invalid_vectors" or "FAIL" in result.stdout


@pytest.mark.parametrize(
    ("filename", "delta"),
    [
        ("input_vectors.txt", 1 << 8),
        ("expected_classes.txt", 2),
        ("expected_logits.txt", 1 << 32),
        ("input_vectors.txt", 1 << 64),
        ("expected_classes.txt", 1 << 64),
        ("expected_logits.txt", 1 << 64),
        ("input_vectors.txt", "x"),
        ("expected_classes.txt", "x"),
        ("expected_logits.txt", "x"),
        ("input_vectors.txt", "z"),
        ("expected_classes.txt", "z"),
        ("expected_logits.txt", "z"),
    ],
)
def test_simulation_rejects_vector_values_that_alias_after_sv_truncation(
    tmp_path: Path,
    filename: str,
    delta: int | str,
) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    path = tmp_path / filename
    tokens = path.read_text().split()
    if isinstance(delta, str):
        tokens[-1] = delta
    else:
        tokens = [str(int(token) + delta) for token in tokens]
    path.write_text("\n".join(tokens) + "\n", encoding="utf-8")

    result = run_simulation(tmp_path)

    assert not result.ok
    assert result.status == "invalid_vectors"
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["simulation"]["status"] == "invalid_vectors"


@pytest.mark.parametrize(
    "filename",
    ["input_vectors.txt", "expected_classes.txt", "expected_logits.txt"],
)
def test_simulation_rejects_noncanonical_decimal_vector_tokens(
    tmp_path: Path,
    filename: str,
) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    path = tmp_path / filename
    tokens = path.read_text(encoding="utf-8").split()
    tokens[0] = "+0"
    path.write_text("\n".join(tokens) + "\n", encoding="utf-8")

    result = run_simulation(tmp_path)

    assert not result.ok
    assert result.status == "invalid_vectors"
    assert "canonical signed decimal" in result.message


@pytest.mark.parametrize(
    ("filename", "delta"),
    [
        ("input_vectors.txt", 1 << 8),
        ("expected_classes.txt", 2),
        ("expected_logits.txt", 1 << 32),
        ("input_vectors.txt", 1 << 64),
        ("expected_classes.txt", 1 << 64),
        ("expected_logits.txt", 1 << 64),
        ("input_vectors.txt", "x"),
        ("expected_classes.txt", "x"),
        ("expected_logits.txt", "x"),
        ("input_vectors.txt", "z"),
        ("expected_classes.txt", "z"),
        ("expected_logits.txt", "z"),
    ],
)
def test_generated_testbench_itself_rejects_values_outside_port_ranges(
    tmp_path: Path,
    filename: str,
    delta: int | str,
) -> None:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog is None or vvp is None:
        pytest.skip("Icarus Verilog is required for direct testbench regression")
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    path = tmp_path / filename
    tokens = path.read_text().split()
    if isinstance(delta, str):
        tokens[-1] = delta
    else:
        tokens = [str(int(token) + delta) for token in tokens]
    path.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    sources = [
        "linear_comb.sv",
        "relu.sv",
        "argmax.sv",
        "top.sv",
        "tb_top.sv",
    ]
    compiled = subprocess.run(
        [iverilog, "-g2012", "-o", "simv-direct", *sources],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr

    simulated = subprocess.run(
        [vvp, "-M", "-", "simv-direct"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert simulated.returncode != 0
    assert "FAIL" in simulated.stdout


def test_vector_contract_parser_ignores_commented_localparam_alias(
    tmp_path: Path,
) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    inputs = tmp_path / "input_vectors.txt"
    tokens = [str(int(token) + 256) for token in inputs.read_text().split()]
    inputs.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    metadata_path = tmp_path / "vectors.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["data_bits"] = 16
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    testbench = tmp_path / "tb_top.sv"
    testbench.write_text(
        "// localparam int DATA_BITS = 16;\n" + testbench.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "visualization.json").unlink()

    result = run_simulation(tmp_path)

    assert not result.ok
    assert result.status == "invalid_vectors"


def test_vector_contract_parser_ignores_strings_that_look_like_comments_or_params(
    tmp_path: Path,
) -> None:
    model = nn.Sequential(nn.Linear(2, 1)).eval()
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.zero_()
    graph = parse_model(model, input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    (tmp_path / "input_vectors.txt").write_text("0 0\n0 0\n", encoding="utf-8")
    (tmp_path / "expected_classes.txt").write_text("0\n0\n", encoding="utf-8")
    (tmp_path / "expected_logits.txt").write_text("0\n0\n", encoding="utf-8")
    testbench = tmp_path / "tb_top.sv"
    text = testbench.read_text(encoding="utf-8")
    text = text.replace(
        "localparam int DATA_BITS = 8;",
        '\n'.join(
            [
                'initial $display("/*");',
                "localparam int DATA_BITS = 4;",
                'initial $display("*/");',
                'initial $display("localparam int DATA_BITS = 8;");',
            ]
        ),
        1,
    )
    testbench.write_text(text, encoding="utf-8")

    result = run_simulation(tmp_path)

    assert not result.ok
    assert result.status == "invalid_vectors"
    assert "header" in result.message


def test_vector_contract_rejects_systemverilog_preprocessor_directives(
    tmp_path: Path,
) -> None:
    model = nn.Sequential(nn.Linear(2, 1)).eval()
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.zero_()
    graph = parse_model(model, input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    (tmp_path / "input_vectors.txt").write_text("0 0\n0 0\n", encoding="utf-8")
    (tmp_path / "expected_classes.txt").write_text("0\n0\n", encoding="utf-8")
    (tmp_path / "expected_logits.txt").write_text("0\n0\n", encoding="utf-8")
    testbench = tmp_path / "tb_top.sv"
    text = testbench.read_text(encoding="utf-8")
    text = text.replace(
        "localparam int DATA_BITS = 8;",
        "\n".join(
            [
                "`define TORCH2RTL_LOCALPARAM localparam",
                "`define TORCH2RTL_DATA_BITS int DATA_BITS = 4",
                "`ifdef TORCH2RTL_INACTIVE",
                "localparam int DATA_BITS = 8;",
                "`else",
                "`TORCH2RTL_LOCALPARAM `TORCH2RTL_DATA_BITS;",
                "`endif",
            ]
        ),
        1,
    )
    testbench.write_text(text, encoding="utf-8")

    result = run_simulation(tmp_path)

    assert not result.ok
    assert result.status == "invalid_vectors"
    assert "preprocessor" in result.message


def test_vector_contract_requires_generated_top_level_parameter_header(
    tmp_path: Path,
) -> None:
    model = nn.Sequential(nn.Linear(2, 1)).eval()
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.zero_()
    graph = parse_model(model, input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    (tmp_path / "input_vectors.txt").write_text("0 0\n0 0\n", encoding="utf-8")
    (tmp_path / "expected_classes.txt").write_text("0\n0\n", encoding="utf-8")
    (tmp_path / "expected_logits.txt").write_text("0\n0\n", encoding="utf-8")
    testbench = tmp_path / "tb_top.sv"
    text = testbench.read_text(encoding="utf-8")
    text = text.replace(
        "localparam int DATA_BITS = 8;",
        "\n".join(
            [
                "parameter int DATA_BITS = 4;",
                "generate",
                "    if (0) begin : inactive_contract",
                "        localparam int DATA_BITS = 8;",
                "    end",
                "endgenerate",
            ]
        ),
        1,
    )
    testbench.write_text(text, encoding="utf-8")

    result = run_simulation(tmp_path)

    assert not result.ok
    assert result.status == "invalid_vectors"
    assert "header" in result.message


def test_vector_contract_rejects_modified_generated_checker_body(
    tmp_path: Path,
) -> None:
    model = nn.Sequential(nn.Linear(2, 1)).eval()
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.zero_()
    graph = parse_model(model, input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    (tmp_path / "input_vectors.txt").write_text("0 0\n0 0\n", encoding="utf-8")
    (tmp_path / "expected_classes.txt").write_text("0\n0\n", encoding="utf-8")
    (tmp_path / "expected_logits.txt").write_text("0\n0\n", encoding="utf-8")
    testbench = tmp_path / "tb_top.sv"
    text = testbench.read_text(encoding="utf-8")
    text = text.replace(
        "endmodule",
        'initial begin $display("PASS vectors=2"); $finish(0); end\nendmodule',
        1,
    )
    testbench.write_text(text, encoding="utf-8")

    result = run_simulation(tmp_path)

    assert not result.ok
    assert result.status == "invalid_vectors"
    assert "checker body" in result.message


def test_vector_contract_rejects_simulation_control_in_dut_source(
    tmp_path: Path,
) -> None:
    model = nn.Sequential(nn.Linear(2, 1)).eval()
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.zero_()
    graph = parse_model(model, input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    (tmp_path / "input_vectors.txt").write_text("0 0\n0 0\n", encoding="utf-8")
    (tmp_path / "expected_classes.txt").write_text("0\n0\n", encoding="utf-8")
    (tmp_path / "expected_logits.txt").write_text("0\n0\n", encoding="utf-8")
    top = tmp_path / "top.sv"
    text = top.read_text(encoding="utf-8")
    top.write_text(
        text.replace(
            "endmodule",
            'initial begin $display("PASS vectors=2"); $finish(0); end\nendmodule',
            1,
        ),
        encoding="utf-8",
    )

    result = run_simulation(tmp_path)

    assert not result.ok
    assert result.status == "invalid_vectors"
    assert "DUT source" in result.message


@pytest.mark.parametrize(
    "injection",
    [
        "defparam DATA_BITS = 16;",
        "wire checker_probe; assign checker_probe = tb_top.status;",
    ],
)
def test_vector_contract_binds_every_rtl_source_to_compile_report(
    tmp_path: Path,
    injection: str,
) -> None:
    model = nn.Sequential(nn.Linear(1, 1)).eval()
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.zero_()
    graph = parse_model(model, input_shape=(1,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=1, seed=4)
    (tmp_path / "input_vectors.txt").write_text("0\n", encoding="utf-8")
    (tmp_path / "expected_classes.txt").write_text("0\n", encoding="utf-8")
    (tmp_path / "expected_logits.txt").write_text("0\n", encoding="utf-8")
    top = tmp_path / "top.sv"
    top.write_text(
        top.read_text(encoding="utf-8").replace(
            "endmodule",
            f"{injection}\nendmodule",
            1,
        ),
        encoding="utf-8",
    )

    result = run_simulation(tmp_path)

    assert not result.ok
    assert result.status == "invalid_vectors"
    assert "integrity" in result.message
    synthesis = run_yosys(tmp_path)
    assert not synthesis.ok
    assert synthesis.status == "invalid_sources"
    assert "integrity" in synthesis.message


def test_vector_contract_binds_testbench_width_to_compile_report(
    tmp_path: Path,
) -> None:
    model = nn.Sequential(nn.Linear(2, 1)).eval()
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.zero_()
    graph = parse_model(model, input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    (tmp_path / "input_vectors.txt").write_text("0 0\n0 0\n", encoding="utf-8")
    (tmp_path / "expected_classes.txt").write_text("0\n0\n", encoding="utf-8")
    (tmp_path / "expected_logits.txt").write_text("0\n0\n", encoding="utf-8")
    metadata_path = tmp_path / "vectors.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["data_bits"] = 16
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    testbench = tmp_path / "tb_top.sv"
    text = testbench.read_text(encoding="utf-8")
    text = text.replace("DATA_BITS = 8", "DATA_BITS = 16", 1)
    text = text.replace("DATA_MIN = -64'sd128", "DATA_MIN = -64'sd32768", 1)
    text = text.replace("DATA_MAX = 64'sd127", "DATA_MAX = 64'sd32767", 1)
    testbench.write_text(text, encoding="utf-8")

    result = run_simulation(tmp_path)

    assert not result.ok
    assert result.status == "invalid_vectors"
    assert "compile report" in result.message


def test_synthesis_result_does_not_depend_on_valid_vector_trace(tmp_path: Path) -> None:
    if shutil.which("yosys") is None:
        pytest.skip("Yosys is required for synthesis regression")
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    (tmp_path / "input_vectors.txt").write_text("", encoding="utf-8")

    result = run_yosys(tmp_path)

    assert result.ok
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["synthesis"]["status"] == "passed"
    manifest = json.loads(
        (tmp_path / "visualization.json").read_text(encoding="utf-8")
    )
    assert manifest["build"]["status"]["synthesis"]["status"] == "passed"
    assert manifest["trace"]["matched"] is False


def test_synthesis_result_survives_structurally_corrupt_visualization(
    tmp_path: Path,
) -> None:
    if shutil.which("yosys") is None:
        pytest.skip("Yosys is required for synthesis regression")
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    manifest_path = tmp_path / "visualization.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["quant"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_yosys(tmp_path)

    assert result.ok
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["synthesis"]["status"] == "passed"


def test_synthesis_rejects_report_width_mismatch_before_yosys(tmp_path: Path) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    report_path = tmp_path / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["quant"]["bits"] = 16
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = run_yosys(tmp_path)

    assert not result.ok
    assert result.status == "invalid_sources"
    assert "compile contract" in result.message


def test_synthesis_rejects_noninteger_report_vector_count(tmp_path: Path) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    report_path = tmp_path / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["vectors"]["count"] = 2.0
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = run_yosys(tmp_path)

    assert not result.ok
    assert result.status == "invalid_sources"
    assert "compile contract" in result.message


_MALFORMED_REPORT_CASES = (
    "invalid-json",
    "array-root",
    "status-only-number",
    "status-number",
    "quant-array",
    "vectors-array",
    "quant-bool",
    "vectors-float",
    "generated-files-number",
    "generated-files-item-number",
    "missing-metrics",
)

_HUGE_JSON_INTEGER = "9" * 5000
_DEEPLY_NESTED_JSON_ARRAY = "[" * 10000 + "0" + "]" * 10000
_EXTREME_JSON_PAYLOADS = (
    pytest.param(_HUGE_JSON_INTEGER, id="huge-integer-root"),
    pytest.param(
        '{"status": ' + _HUGE_JSON_INTEGER + "}",
        id="huge-integer-field",
    ),
    pytest.param(_DEEPLY_NESTED_JSON_ARRAY, id="deep-array-root"),
)


def _write_malformed_report(report_path: Path, case: str) -> None:
    if case == "invalid-json":
        report_path.write_text("{not json", encoding="utf-8")
        return
    if case == "array-root":
        report_path.write_text("[]", encoding="utf-8")
        return
    if case == "status-only-number":
        report_path.write_text('{"status": 1}', encoding="utf-8")
        return

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if case == "status-number":
        report["status"] = 1
    elif case == "quant-array":
        report["quant"] = []
    elif case == "vectors-array":
        report["vectors"] = []
    elif case == "quant-bool":
        report["quant"]["bits"] = True
    elif case == "vectors-float":
        report["vectors"]["count"] = 2.0
    elif case == "generated-files-number":
        report["generated_files"] = 1
    elif case == "generated-files-item-number":
        report["generated_files"] = ["top.sv", 1]
    elif case == "missing-metrics":
        del report["metrics"]
    else:
        raise AssertionError(f"unknown malformed report case: {case}")
    report_path.write_text(json.dumps(report), encoding="utf-8")


@pytest.mark.parametrize("case", _MALFORMED_REPORT_CASES)
def test_verify_rejects_malformed_compile_report_before_eda_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    case: str,
) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    _write_malformed_report(tmp_path / "report.json", case)

    def unexpected_eda_lookup(_name: str) -> str | None:
        pytest.fail("verify reached EDA discovery after failed report preflight")

    monkeypatch.setattr("torch2rtl.verify.simulator.find_eda_tool", unexpected_eda_lookup)

    result = run_simulation(tmp_path)
    rc = cli.cmd_verify(argparse.Namespace(build_dir=tmp_path))
    output = capsys.readouterr().out

    assert not result.ok
    assert "invalid compile report" in result.message
    assert rc == 1
    assert "invalid compile report" in output
    assert "Traceback" not in output
    recorded = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert recorded["preflight_failure"]["section"] == "simulation"
    assert recorded["status"]["simulation"]["ok"] is False


@pytest.mark.parametrize("case", _MALFORMED_REPORT_CASES)
def test_synth_rejects_malformed_compile_report_before_eda_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    case: str,
) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    _write_malformed_report(tmp_path / "report.json", case)

    def unexpected_eda_lookup(_name: str) -> str | None:
        pytest.fail("synth reached EDA discovery after failed report preflight")

    monkeypatch.setattr("torch2rtl.synth.yosys.find_eda_tool", unexpected_eda_lookup)

    result = run_yosys(tmp_path)
    rc = cli.cmd_synth(argparse.Namespace(build_dir=tmp_path))
    output = capsys.readouterr().out

    assert not result.ok
    assert "invalid compile report" in result.message
    assert rc == 1
    assert "invalid compile report" in output
    assert "Traceback" not in output
    recorded = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert recorded["preflight_failure"]["section"] == "synthesis"
    assert recorded["status"]["synthesis"]["ok"] is False


@pytest.mark.parametrize("section", ["simulation", "synthesis"])
@pytest.mark.parametrize(
    "malformed_field",
    ["status", "generated-files-number", "generated-files-item-number"],
)
def test_malformed_report_replaces_stale_pass_in_report_and_visualization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    malformed_field: str,
) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    if section == "simulation":
        passed: dict[str, object] = SimulationResult(
            True,
            "passed",
            "simulation passed",
            stdout="PASS vectors=2\n",
        ).__dict__
    else:
        passed = SynthResult(
            True,
            "passed",
            "synthesis passed",
            stdout="Number of cells: 1\n",
        ).__dict__
    update_report(tmp_path, section, passed)

    report_path = tmp_path / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if malformed_field == "status":
        report["status"] = 1
    elif malformed_field == "generated-files-number":
        report["generated_files"] = 1
    else:
        report["generated_files"] = ["top.sv", 1]
    report_path.write_text(json.dumps(report), encoding="utf-8")

    if section == "simulation":
        monkeypatch.setattr(
            "torch2rtl.verify.simulator.find_eda_tool",
            lambda _name: pytest.fail("verify reached EDA after failed preflight"),
        )
        result = run_simulation(tmp_path)
    else:
        monkeypatch.setattr(
            "torch2rtl.synth.yosys.find_eda_tool",
            lambda _name: pytest.fail("synth reached EDA after failed preflight"),
        )
        result = run_yosys(tmp_path)

    assert not result.ok
    repaired_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert repaired_report["preflight_failure"]["section"] == section
    assert repaired_report["status"][section]["ok"] is False
    assert repaired_report[section]["ok"] is False

    manifest = json.loads(
        (tmp_path / "visualization.json").read_text(encoding="utf-8")
    )
    assert manifest["build"]["status"][section]["ok"] is False
    assert manifest["build"][section]["ok"] is False
    html = (tmp_path / "visualization.html").read_text(encoding="utf-8")
    assert f'"message": "{section} passed"' not in html
    if section == "simulation":
        assert '"stdout": "PASS vectors=2\\n"' not in html


@pytest.mark.parametrize("section", ["simulation", "synthesis"])
@pytest.mark.parametrize("payload_text", _EXTREME_JSON_PAYLOADS)
def test_extreme_compile_report_json_is_controlled_before_eda_for_api_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    section: str,
    payload_text: str,
) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    if section == "simulation":
        passed: dict[str, object] = SimulationResult(
            True,
            "passed",
            "simulation passed",
            stdout="PASS vectors=2\n",
        ).__dict__
        monkeypatch.setattr(
            "torch2rtl.verify.simulator.find_eda_tool",
            lambda _name: pytest.fail("verify reached EDA after failed preflight"),
        )
        run_api = run_simulation
        run_cli = lambda: cli.cmd_verify(argparse.Namespace(build_dir=tmp_path))
    else:
        passed = SynthResult(
            True,
            "passed",
            "synthesis passed",
            stdout="Number of cells: 1\n",
        ).__dict__
        monkeypatch.setattr(
            "torch2rtl.synth.yosys.find_eda_tool",
            lambda _name: pytest.fail("synth reached EDA after failed preflight"),
        )
        run_api = run_yosys
        run_cli = lambda: cli.cmd_synth(argparse.Namespace(build_dir=tmp_path))
    update_report(tmp_path, section, passed)

    report_path = tmp_path / "report.json"
    report_path.write_text(payload_text, encoding="utf-8")
    result = run_api(tmp_path)

    assert not result.ok
    assert "invalid compile report" in result.message
    recorded = json.loads(report_path.read_text(encoding="utf-8"))
    assert recorded["preflight_failure"]["section"] == section
    assert recorded["status"][section]["ok"] is False
    assert recorded[section]["ok"] is False

    manifest = json.loads(
        (tmp_path / "visualization.json").read_text(encoding="utf-8")
    )
    assert manifest["build"]["status"][section]["ok"] is False
    assert manifest["build"][section]["ok"] is False
    html = (tmp_path / "visualization.html").read_text(encoding="utf-8")
    assert f'"message": "{section} passed"' not in html
    assert '"stdout": "PASS vectors=2\\n"' not in html

    report_path.write_text(payload_text, encoding="utf-8")
    rc = run_cli()
    output = capsys.readouterr().out

    assert rc == 1
    assert "invalid compile report" in output
    assert "Traceback" not in output
    recorded = json.loads(report_path.read_text(encoding="utf-8"))
    assert recorded["preflight_failure"]["section"] == section
    assert recorded["status"][section]["ok"] is False


@pytest.mark.parametrize("section", ["simulation", "synthesis"])
@pytest.mark.parametrize("payload_text", _EXTREME_JSON_PAYLOADS)
def test_extreme_vectors_json_is_controlled_before_eda_for_api_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    section: str,
    payload_text: str,
) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    if section == "simulation":
        passed: dict[str, object] = SimulationResult(
            True,
            "passed",
            "simulation passed",
            stdout="PASS vectors=2\n",
        ).__dict__
        expected_message = "invalid vector metadata"
        monkeypatch.setattr(
            "torch2rtl.verify.simulator.find_eda_tool",
            lambda _name: pytest.fail("verify reached EDA after failed preflight"),
        )
        run_api = run_simulation
        run_cli = lambda: cli.cmd_verify(argparse.Namespace(build_dir=tmp_path))
    else:
        passed = SynthResult(
            True,
            "passed",
            "synthesis passed",
            stdout="Number of cells: 1\n",
        ).__dict__
        expected_message = "invalid synthesis compile contract"
        monkeypatch.setattr(
            "torch2rtl.synth.yosys.find_eda_tool",
            lambda _name: pytest.fail("synth reached EDA after failed preflight"),
        )
        run_api = run_yosys
        run_cli = lambda: cli.cmd_synth(argparse.Namespace(build_dir=tmp_path))
    update_report(tmp_path, section, passed)

    vectors_path = tmp_path / "vectors.json"
    vectors_path.write_text(payload_text, encoding="utf-8")
    result = run_api(tmp_path)

    assert not result.ok
    assert expected_message in result.message
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["status"][section]["ok"] is False
    assert report[section]["ok"] is False
    manifest = json.loads(
        (tmp_path / "visualization.json").read_text(encoding="utf-8")
    )
    assert manifest["build"]["status"][section]["ok"] is False
    assert manifest["build"][section]["ok"] is False
    html = (tmp_path / "visualization.html").read_text(encoding="utf-8")
    assert f'"message": "{section} passed"' not in html
    assert '"stdout": "PASS vectors=2\\n"' not in html

    rc = run_cli()
    output = capsys.readouterr().out

    assert rc == 1
    assert expected_message in output
    assert "Traceback" not in output
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["status"][section]["ok"] is False


@pytest.mark.parametrize(
    ("filename", "declaration", "replacement", "verify_message"),
    [
        (
            "tb_top.sv",
            "localparam int DATA_BITS = 8;",
            f"localparam int DATA_BITS = {_HUGE_JSON_INTEGER};",
            "invalid testbench contract",
        ),
        (
            "top.sv",
            "parameter int DATA_BITS = 8,",
            f"parameter int DATA_BITS = {_HUGE_JSON_INTEGER},",
            "invalid compiled top contract",
        ),
    ],
    ids=["testbench-localparam", "top-parameter"],
)
def test_extreme_systemverilog_contract_integer_is_controlled_before_eda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    declaration: str,
    replacement: str,
    verify_message: str,
) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    source_path = tmp_path / filename
    source_path.write_text(
        source_path.read_text(encoding="utf-8").replace(declaration, replacement, 1),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["rtl_sha256"][filename] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(
        "torch2rtl.verify.simulator.find_eda_tool",
        lambda _name: pytest.fail("verify reached EDA after failed RTL preflight"),
    )
    monkeypatch.setattr(
        "torch2rtl.synth.yosys.find_eda_tool",
        lambda _name: pytest.fail("synth reached EDA after failed RTL preflight"),
    )

    verification = run_simulation(tmp_path)
    synthesis = run_yosys(tmp_path)

    assert not verification.ok
    assert verification.status == "invalid_vectors"
    assert verify_message in verification.message
    assert not synthesis.ok
    assert synthesis.status == "invalid_sources"
    assert "invalid synthesis compile contract" in synthesis.message


def test_vector_whitespace_layout_does_not_break_post_simulation_refresh(
    tmp_path: Path,
) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    path = tmp_path / "input_vectors.txt"
    path.write_text("\n".join(path.read_text().split()) + "\n", encoding="utf-8")

    result = run_simulation(tmp_path)

    assert result.ok
    manifest = json.loads(
        (tmp_path / "visualization.json").read_text(encoding="utf-8")
    )
    assert manifest["build"]["status"]["simulation"]["status"] == "passed"


def test_invalid_vectors_replace_stale_visualization_pass_status(tmp_path: Path) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    assert run_simulation(tmp_path).ok
    (tmp_path / "input_vectors.txt").write_text("", encoding="utf-8")

    result = run_simulation(tmp_path)

    assert not result.ok
    assert result.status == "invalid_vectors"
    manifest = json.loads(
        (tmp_path / "visualization.json").read_text(encoding="utf-8")
    )
    assert manifest["build"]["status"]["simulation"]["status"] == "invalid_vectors"
    assert manifest["trace"]["matched"] is False
    html = (tmp_path / "visualization.html").read_text(encoding="utf-8")
    assert '"status": "invalid_vectors"' in html
    assert '"message": "simulation passed"' not in html


def test_simulation_rejects_non_object_vector_metadata_and_records_failure(
    tmp_path: Path,
) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    (tmp_path / "vectors.json").write_text("[]", encoding="utf-8")

    result = run_simulation(tmp_path)

    assert not result.ok
    assert result.status == "invalid_vectors"
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["simulation"]["status"] == "invalid_vectors"


@pytest.mark.parametrize(
    "filename",
    [
        "vectors.json",
        "input_vectors.txt",
        "expected_classes.txt",
        "expected_logits.txt",
    ],
)
def test_simulation_rejects_invalid_utf8_vector_artifacts(
    tmp_path: Path,
    filename: str,
) -> None:
    graph = parse_model(nn.Sequential(nn.Linear(2, 2)), input_shape=(2,))
    emit_systemverilog(graph, FixedPointConfig(), tmp_path, vector_count=2, seed=4)
    (tmp_path / filename).write_bytes(b"\xff\xfe")

    result = run_simulation(tmp_path)

    assert not result.ok
    assert result.status == "invalid_vectors"


def test_simulation_pass_marker_requires_nonzero_exact_vector_count() -> None:
    assert not _simulation_ok(0, "PASS vectors=0\n")
    assert not _simulation_ok(0, "PASS vectors=1\n", expected_vectors=2)
    assert _simulation_ok(0, "PASS vectors=2\n", expected_vectors=2)


def test_generated_conv_rtl_matches_python_on_directed_boundaries(
    tmp_path: Path,
) -> None:
    model = nn.Sequential(
        nn.Conv2d(1, 3, kernel_size=2, stride=2, padding=1),
    ).eval()
    with torch.no_grad():
        model[0].weight.copy_(
            torch.tensor(
                [
                    [[[127 / 64, 127 / 64], [127 / 64, 127 / 64]]],
                    [[[-2.0, -2.0], [-2.0, -2.0]]],
                    [[[1.0, -1.0], [1.0, -1.0]]],
                ]
            )
        )
        model[0].bias.copy_(torch.tensor([127 / 64, -2.0, 0.0]))
    graph = parse_model(model, input_shape=(1, 3, 3))
    cfg = FixedPointConfig(bits=8, frac_bits=6, acc_bits=24)
    inputs = np.asarray(
        [
            [127] * 9,
            [-128] * 9,
            [127, -128, 0, -1, 1, 64, -64, 126, -127],
            [0] * 9,
            [-128, 127, -128, 127, -128, 127, -128, 127, -128],
        ],
        dtype=np.int64,
    )
    emit_systemverilog(graph, cfg, tmp_path, vector_count=len(inputs), seed=0)
    qgraph = quantize_graph(graph, cfg)
    results = [infer_quantized(qgraph, row) for row in inputs]
    logits = np.stack([result.logits.reshape(-1) for result in results])
    classes = np.asarray([result.class_id for result in results], dtype=np.int64)
    np.savetxt(tmp_path / "input_vectors.txt", inputs, fmt="%d")
    np.savetxt(tmp_path / "expected_logits.txt", logits, fmt="%d")
    np.savetxt(tmp_path / "expected_classes.txt", classes.reshape(-1, 1), fmt="%d")

    simulation = run_simulation(tmp_path)

    assert simulation.ok, simulation.stdout + simulation.stderr
    assert "PASS vectors=5" in simulation.stdout
    assert np.any(logits == cfg.min_int)
    assert np.any(logits == cfg.max_int)


def test_signed_32_bit_data_endpoints_match_generated_rtl(tmp_path: Path) -> None:
    model = nn.Sequential(nn.Linear(1, 1, bias=False)).eval()
    with torch.no_grad():
        model[0].weight.fill_(1.0)
    cfg = FixedPointConfig(bits=32, frac_bits=0, acc_bits=64)
    graph = parse_model(model, input_shape=(1,))
    inputs = np.asarray([[cfg.min_int], [cfg.max_int]], dtype=np.int64)
    emit_systemverilog(graph, cfg, tmp_path, vector_count=2, seed=0)
    qgraph = quantize_graph(graph, cfg)
    results = [infer_quantized(qgraph, row) for row in inputs]
    logits = np.stack([result.logits.reshape(-1) for result in results])
    classes = np.asarray([result.class_id for result in results], dtype=np.int64)
    np.savetxt(tmp_path / "input_vectors.txt", inputs, fmt="%d")
    np.savetxt(tmp_path / "expected_logits.txt", logits, fmt="%d")
    np.savetxt(tmp_path / "expected_classes.txt", classes.reshape(-1, 1), fmt="%d")

    simulation = run_simulation(tmp_path)

    assert simulation.ok, simulation.stdout + simulation.stderr
    assert "PASS vectors=2" in simulation.stdout
    np.testing.assert_array_equal(logits.reshape(-1), [cfg.min_int, cfg.max_int])


def test_explicit_verify_and_synth_fail_when_tool_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        cli,
        "run_simulation",
        lambda _path: SimulationResult(False, "not_found", "simulator not found"),
    )
    monkeypatch.setattr(
        cli,
        "run_yosys",
        lambda _path: SynthResult(False, "not_found", "yosys not found"),
    )

    assert cli.cmd_verify(argparse.Namespace(build_dir=tmp_path)) == 1
    assert cli.cmd_synth(argparse.Namespace(build_dir=tmp_path)) == 1


def test_pytest_configuration_does_not_require_repo_local_temp_directory() -> None:
    config = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "--basetemp=temp/pytest" not in config


def test_tiny_mlp_real_train_checkpoint_compile_path(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny_mlp.pt"
    out_dir = tmp_path / "rtl"
    train = subprocess.run(
        [
            sys.executable,
            "examples/tiny_mlp/train.py",
            "--epochs",
            "1",
            "--out",
            str(checkpoint),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert train.returncode == 0, train.stdout + train.stderr
    compile_result = subprocess.run(
        [
            sys.executable,
            "examples/tiny_mlp/compile.py",
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(out_dir),
            "--vectors",
            "2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr
    assert checkpoint.exists()
    assert (out_dir / "top.sv").exists()
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report["vectors"]["count"] == 2
    assert str(checkpoint) in compile_result.stdout

    trained_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    trained_weight = trained_state["net.0.weight"].detach().numpy()
    torch.manual_seed(0)
    from examples.tiny_mlp.model import create_model

    initial_weight = create_model().net[0].weight.detach().numpy()
    cfg = FixedPointConfig()
    trained_q = quantize_array(trained_weight, cfg).reshape(-1)
    initial_q = quantize_array(initial_weight, cfg).reshape(-1)
    changed = np.flatnonzero(trained_q != initial_q)
    assert changed.size > 0
    weight_index = int(changed[0])
    trained_value = int(trained_q[weight_index])
    literal = (
        f"-{cfg.bits}'sd{abs(trained_value)}"
        if trained_value < 0
        else f"{cfg.bits}'sd{trained_value}"
    )
    top = (out_dir / "top.sv").read_text(encoding="utf-8")
    assert (
        f"NET_0_0_WEIGHTS[{weight_index}*DATA_BITS +: DATA_BITS] = {literal};"
        in top
    )


def test_wheel_configuration_includes_runtime_demo_models() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    force_include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"][
        "force-include"
    ]
    required = {
        "examples/tiny_mlp/model.py",
        "examples/grid_classifier/model.py",
        "examples/tiny_conv/model.py",
        "examples/image_cnn/model.py",
        "examples/image_cnn/data.py",
    }

    assert required <= set(force_include)
    assert all(force_include[path] == path for path in required)


def test_v02_python_boundary_is_synchronized_across_release_sources() -> None:
    required_python = ">=3.12,<3.13"
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads(Path("uv.lock").read_text(encoding="utf-8"))

    assert pyproject["project"]["requires-python"] == required_python
    assert lock["requires-python"] == required_python

    release_docs = {
        "README.md": "Python `>=3.12,<3.13`",
        "docs/v0.2_semantic_correctness.md": "Python `>=3.12,<3.13`",
        "docs/torch2rtl_guide.md": "Python `>=3.12,<3.13`",
    }
    for filename, required_text in release_docs.items():
        text = Path(filename).read_text(encoding="utf-8")
        assert required_text in text, filename
        assert "Python 3.11+" not in text, filename
        assert "Python `3.11+`" not in text, filename
        assert "Python 3.12+" not in text, filename
        assert "Python `3.12+`" not in text, filename

    readme = Path("README.md").read_text(encoding="utf-8")
    assert "badge/Python-3.12-" in readme
    assert "badge/Python-3.11" not in readme


def test_v02_integrity_documentation_states_local_manifest_threat_boundary() -> None:
    for filename in (
        "README.md",
        "docs/v0.2_semantic_correctness.md",
        "docs/torch2rtl_guide.md",
    ):
        text = Path(filename).read_text(encoding="utf-8")
        normalized = " ".join(text.split())
        assert "не является внешним корнем доверия" in normalized, filename
        assert "согласован" in text and "вне" in text, filename
        assert "threat model" in text or "модели защиты" in text, filename
        assert "внешн" in normalized and "подпис" in normalized, filename
        assert "доверенн" in normalized and "manifest" in normalized, filename
        assert "не гарант" in normalized, filename
        assert "произволь" in normalized and "vector payload" in normalized, filename
