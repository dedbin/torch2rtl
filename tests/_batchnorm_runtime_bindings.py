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



def test_rejects_monkeypatched_official_fusion_without_calling_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False
    original = torch.nn.utils.fusion.fuse_conv_bn_eval

    def replacement(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.nn.utils.fusion, "fuse_conv_bn_eval", replacement)

    with pytest.raises(UnsupportedOpError, match="framework|fusion|modified"):
        parse_model(_basic_model(), input_shape=(1, 2, 2))
    assert not called


def test_rejects_monkeypatched_batchnorm_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = nn.BatchNorm2d.forward

    def replacement(self: nn.BatchNorm2d, inputs: torch.Tensor) -> torch.Tensor:
        return original(self, inputs) + 1.0

    monkeypatch.setattr(nn.BatchNorm2d, "forward", replacement)

    with pytest.raises(UnsupportedOpError, match="framework|modified"):
        parse_model(_basic_model(), input_shape=(1, 2, 2))


def test_rejects_monkeypatched_tensor_dim_before_executing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _basic_model()
    called = False

    def replacement(_tensor: torch.Tensor) -> int:
        nonlocal called
        called = True
        raise AssertionError("untrusted Tensor.dim must not execute")

    monkeypatch.setattr(torch.Tensor, "dim", replacement)

    with pytest.raises(UnsupportedOpError) as error:
        parse_model(model, input_shape=(1, 2, 2))
    assert not called
    assert "dim" in str(error.value)


@pytest.mark.parametrize("name", ["isfinite", "all", "allclose", "argmax"])
def test_rejects_monkeypatched_numpy_semantic_binding_before_executing_it(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    model = _basic_model()
    called = False

    def replacement(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError(f"untrusted numpy.{name} must not execute")

    monkeypatch.setattr(np, name, replacement)

    with pytest.raises(UnsupportedOpError) as error:
        parse_model(model, input_shape=(1, 2, 2))
    assert not called
    assert name in str(error.value)


def test_rejects_replaced_frontend_numpy_module_without_accessing_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _basic_model()
    proxy_called = False

    class NumpyProxy:
        def __getattr__(self, name: str) -> object:
            nonlocal proxy_called
            proxy_called = True
            raise AssertionError(f"untrusted NumPy proxy attribute accessed: {name}")

    monkeypatch.setattr(fx_frontend, "np", NumpyProxy())

    with pytest.raises(UnsupportedOpError, match="NumPy module binding"):
        fx_frontend.parse_model(model, input_shape=(1, 2, 2))
    assert not proxy_called


@pytest.mark.parametrize(
    ("owner_name", "attribute_name"),
    [
        pytest.param("conv", "weight", id="conv-weight"),
        pytest.param("conv", "bias", id="conv-bias"),
        pytest.param("batchnorm", "weight", id="batchnorm-weight"),
        pytest.param("batchnorm", "bias", id="batchnorm-bias"),
        pytest.param("batchnorm", "running_mean", id="batchnorm-running-mean"),
        pytest.param("batchnorm", "running_var", id="batchnorm-running-var"),
        pytest.param(
            "batchnorm",
            "num_batches_tracked",
            id="batchnorm-num-batches-tracked",
        ),
    ],
)
def test_rejects_parameter_and_buffer_registry_slots_shadowed_by_instance_state(
    owner_name: str,
    attribute_name: str,
) -> None:
    model = _basic_model()
    owner = model[0] if owner_name == "conv" else model[1]
    object.__setattr__(owner, attribute_name, None)

    with pytest.raises(
        UnsupportedOpError,
        match=f"{owner_name.replace('batchnorm', 'BatchNorm2d').replace('conv', 'Conv2d')}"
        ".*(shadow|schema)|shadow.*(Conv2d|BatchNorm2d)",
    ):
        parse_model(model, input_shape=(1, 2, 2))


@pytest.mark.parametrize(
    "module_name",
    ["builtins", "copy", "dis", "inspect", "math"],
)
def test_rejects_replaced_frontend_runtime_module_without_accessing_proxy(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
) -> None:
    model = _basic_model()
    proxy_calls = 0

    class RuntimeProxy:
        def __getattr__(self, name: str) -> object:
            nonlocal proxy_calls
            proxy_calls += 1
            raise AssertionError(
                f"untrusted {module_name} proxy attribute accessed: {name}"
            )

    monkeypatch.setattr(fx_frontend, module_name, RuntimeProxy())

    with pytest.raises(
        UnsupportedOpError,
        match=f"{module_name}.*module binding|module binding.*{module_name}",
    ):
        fx_frontend.parse_model(model, input_shape=(1, 2, 2))
    assert proxy_calls == 0


@pytest.mark.parametrize(
    ("owner", "attribute_name"),
    [
        pytest.param(copy, "deepcopy", id="copy-deepcopy"),
        pytest.param(inspect, "getattr_static", id="inspect-getattr-static"),
        pytest.param(inspect, "unwrap", id="inspect-unwrap"),
        pytest.param(dis, "Bytecode", id="dis-bytecode"),
        pytest.param(nn, "Parameter", id="nn-parameter"),
    ],
)
def test_missing_trusted_runtime_binding_is_a_controlled_error(
    monkeypatch: pytest.MonkeyPatch,
    owner: object,
    attribute_name: str,
) -> None:
    model = _basic_model()
    with monkeypatch.context() as scoped:
        scoped.delattr(owner, attribute_name)
        caught: UnsupportedOpError | None = None
        try:
            parse_model(model, input_shape=(1, 2, 2))
        except UnsupportedOpError as exc:
            caught = exc

    assert caught is not None
    assert attribute_name.lower() in str(caught).lower()
