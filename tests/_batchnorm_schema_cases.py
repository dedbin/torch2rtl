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



def test_rejects_batchnorm_without_running_statistics() -> None:
    model = nn.Sequential(
        nn.Conv2d(1, 1, 1),
        nn.BatchNorm2d(1, track_running_stats=False),
    ).eval()

    with pytest.raises(UnsupportedOpError, match="track_running_stats|running"):
        parse_model(model, input_shape=(1, 2, 2))


@pytest.mark.parametrize("name", ["running_mean", "running_var"])
def test_rejects_missing_running_statistic(name: str) -> None:
    model = _basic_model()
    setattr(model[1], name, None)

    with pytest.raises(UnsupportedOpError, match=name):
        parse_model(model, input_shape=(1, 2, 2))


def test_rejects_batchnorm_channel_mismatch() -> None:
    model = nn.Sequential(
        nn.Conv2d(1, 2, 1),
        nn.BatchNorm2d(3),
    ).eval()

    with pytest.raises(UnsupportedOpError, match="channel|num_features|shape"):
        parse_model(model, input_shape=(1, 2, 2))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        pytest.param("running_mean", torch.zeros(1, 1), id="mean-rank"),
        pytest.param("running_var", torch.ones(2), id="variance-length"),
        pytest.param("weight", nn.Parameter(torch.ones(2)), id="gamma-length"),
        pytest.param("bias", nn.Parameter(torch.zeros(1, 1)), id="beta-rank"),
    ],
)
def test_rejects_corrupted_batchnorm_shapes(name: str, value: torch.Tensor) -> None:
    model = _basic_model()
    setattr(model[1], name, value)

    with pytest.raises(UnsupportedOpError, match=f"{name}|shape|num_features"):
        parse_model(model, input_shape=(1, 2, 2))


def test_rejects_missing_batchnorm_weight_when_affine_is_true() -> None:
    model = _basic_model()
    model[1].weight = None
    before = _source_snapshot(model)

    with pytest.raises(UnsupportedOpError, match="BatchNorm2d.*weight|weight.*Parameter"):
        parse_model(model, input_shape=(1, 2, 2))

    assert _source_snapshot(model) == before


@pytest.mark.parametrize("name", ["weight", "bias"])
def test_rejects_injected_affine_parameter_when_affine_is_false(name: str) -> None:
    model = _basic_model(bn_affine=False)
    setattr(model[1], name, nn.Parameter(torch.ones(1)))
    before = _source_snapshot(model)

    with pytest.raises(UnsupportedOpError, match="affine=False|parameter schema"):
        parse_model(model, input_shape=(1, 2, 2))

    assert _source_snapshot(model) == before


def test_rejects_non_integer_batchnorm_num_features() -> None:
    model = _basic_model()
    model[1].num_features = 1.0

    with pytest.raises(UnsupportedOpError, match="num_features|integer"):
        parse_model(model, input_shape=(1, 2, 2))


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize(
    "field",
    [
        "conv.weight",
        "conv.bias",
        "bn.weight",
        "bn.bias",
        "bn.running_mean",
        "bn.running_var",
    ],
)
def test_rejects_nonfinite_fusion_inputs(field: str, bad_value: float) -> None:
    model = _basic_model()
    owner_name, attribute = field.split(".")
    owner = model[0] if owner_name == "conv" else model[1]
    tensor = getattr(owner, attribute)
    assert isinstance(tensor, torch.Tensor)
    with torch.no_grad():
        tensor.reshape(-1)[0] = bad_value

    with pytest.raises(UnsupportedOpError, match="finite|NaN|Inf"):
        parse_model(model, input_shape=(1, 2, 2))


def test_rejects_negative_running_variance_even_when_sum_with_eps_is_positive() -> None:
    model = _basic_model()
    model[1].eps = 1.0
    assert model[1].running_var is not None
    model[1].running_var.fill_(-0.5)

    with pytest.raises(UnsupportedOpError, match="running_var|variance"):
        parse_model(model, input_shape=(1, 2, 2))


@pytest.mark.parametrize(
    ("variance", "eps"),
    [
        pytest.param(0.0, 0.0, id="exact-zero"),
        pytest.param(0.0, 1.0e-50, id="float32-underflow-to-zero"),
    ],
)
def test_rejects_nonpositive_running_variance_plus_eps(
    variance: float,
    eps: float,
) -> None:
    model = _basic_model()
    assert model[1].running_var is not None
    model[1].running_var.fill_(variance)
    model[1].eps = eps

    with pytest.raises(UnsupportedOpError, match="eps|positive|running_var"):
        parse_model(model, input_shape=(1, 2, 2))


@pytest.mark.parametrize("eps", [math.nan, math.inf, -math.inf, True, "1e-5"])
def test_rejects_invalid_batchnorm_eps(eps: object) -> None:
    model = _basic_model()
    model[1].eps = eps

    with pytest.raises(UnsupportedOpError, match="eps|finite|float"):
        parse_model(model, input_shape=(1, 2, 2))


def test_rejects_finite_negative_eps_before_official_fusion() -> None:
    model = _basic_model()
    assert model[1].running_var is not None
    model[1].running_var.fill_(1.0)
    model[1].eps = -0.25
    before = _source_snapshot(model)

    with pytest.raises(
        UnsupportedOpError,
        match="eps.*finite non-negative|finite non-negative.*eps",
    ) as error:
        parse_model(model, input_shape=(1, 2, 2))

    assert error.value.__cause__ is None
    assert _source_snapshot(model) == before


def test_rejects_grouped_conv_batchnorm_pair() -> None:
    model = nn.Sequential(
        nn.Conv2d(2, 2, kernel_size=1, groups=2),
        nn.BatchNorm2d(2),
    ).eval()

    with pytest.raises(UnsupportedOpError, match="groups|grouped"):
        parse_model(model, input_shape=(2, 2, 2))


def test_rejects_nonfinite_materialized_fused_weight() -> None:
    model = _basic_model()
    with torch.no_grad():
        model[0].weight.fill_(torch.finfo(torch.float32).max)
        assert model[1].weight is not None
        assert model[1].running_var is not None
        model[1].weight.fill_(4.0)
        model[1].running_var.fill_(3.0)
        model[1].eps = 1.0

    with pytest.raises(UnsupportedOpError, match="fused|weight|finite"):
        parse_model(model, input_shape=(1, 2, 2))


def test_rejects_mixed_conv_batchnorm_dtype() -> None:
    model = nn.Sequential(
        nn.Conv2d(1, 1, 1, dtype=torch.float32),
        nn.BatchNorm2d(1, dtype=torch.float64),
    ).eval()

    with pytest.raises(UnsupportedOpError, match="mixed|dtype"):
        parse_model(model, input_shape=(1, 2, 2))


class _ConvSubclass(nn.Conv2d):
    pass


class _BatchNormSubclass(nn.BatchNorm2d):
    pass


@pytest.mark.parametrize(
    "model",
    [
        nn.Sequential(_ConvSubclass(1, 1, 1), nn.BatchNorm2d(1)).eval(),
        nn.Sequential(nn.Conv2d(1, 1, 1), _BatchNormSubclass(1)).eval(),
    ],
    ids=["conv-subclass", "batchnorm-subclass"],
)
def test_rejects_conv_and_batchnorm_subclasses(model: nn.Module) -> None:
    with pytest.raises(UnsupportedOpError, match="subclass|type"):
        parse_model(model, input_shape=(1, 2, 2))


@pytest.mark.parametrize("hook_kind", ["forward", "forward-pre", "backward"])
def test_rejects_batchnorm_hooks(hook_kind: str) -> None:
    model = _basic_model()
    bn = model[1]
    if hook_kind == "forward":
        bn.register_forward_hook(lambda _module, _args, output: output)
    elif hook_kind == "forward-pre":
        bn.register_forward_pre_hook(lambda _module, args: args)
    else:
        bn.register_full_backward_hook(
            lambda _module, grad_input, grad_output: grad_input
        )
    before = _source_snapshot(model)

    with pytest.raises(UnsupportedOpError, match="hook"):
        parse_model(model, input_shape=(1, 2, 2))
    assert _source_snapshot(model) == before
