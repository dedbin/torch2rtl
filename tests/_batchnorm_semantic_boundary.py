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
from torch2rtl.frontend import _semantics as semantics
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




@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_huge_finite_variance_and_eps_respect_numpy_seterr_and_fail_cleanly(
    dtype: torch.dtype,
) -> None:
    model = _basic_model(dtype=dtype)
    assert model[1].running_var is not None
    largest = torch.finfo(dtype).max
    with torch.no_grad():
        model[1].running_var.fill_(largest)
    model[1].eps = float(largest)

    previous = np.seterr(all="raise")
    caught: UnsupportedOpError | None = None
    observed: dict[str, str] | None = None
    try:
        try:
            parse_model(model, input_shape=(1, 2, 2), input_dtype=dtype)
        except UnsupportedOpError as exc:
            caught = exc
        observed = np.geterr().copy()
    finally:
        np.seterr(**previous)

    assert caught is not None
    assert "running_var" in str(caught) or "finite" in str(caught)
    assert observed == {
        "divide": "raise",
        "over": "raise",
        "under": "raise",
        "invalid": "raise",
    }


@pytest.mark.parametrize(
    ("owner", "name"),
    [
        pytest.param("root", "num_batches_tracked", id="same-name-on-root"),
        pytest.param("conv", "num_batches_tracked", id="same-name-on-conv"),
        pytest.param("batchnorm", "rogue_counter", id="rogue-on-batchnorm"),
    ],
)
def test_int64_buffer_exception_is_limited_to_exact_batchnorm_slot(
    owner: str,
    name: str,
) -> None:
    model = _basic_model()
    target = model if owner == "root" else model[0] if owner == "conv" else model[1]
    target.register_buffer(name, torch.tensor(0, dtype=torch.int64))

    with pytest.raises(UnsupportedOpError, match="int64|dtype|buffer"):
        parse_model(model, input_shape=(1, 2, 2))


class _CounterTensor(torch.Tensor):
    pass


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(torch.tensor([0], dtype=torch.int64), id="non-scalar"),
        pytest.param(torch.tensor(0.0, dtype=torch.float32), id="float32"),
        pytest.param(torch.tensor(False, dtype=torch.bool), id="bool"),
        pytest.param(
            torch.tensor(0, dtype=torch.int64).as_subclass(_CounterTensor),
            id="tensor-subclass",
        ),
        pytest.param(torch.empty((), dtype=torch.int64, device="meta"), id="meta-device"),
    ],
)
def test_rejects_malformed_num_batches_tracked(value: torch.Tensor) -> None:
    model = _basic_model()
    model[1].num_batches_tracked = value

    with pytest.raises(
        UnsupportedOpError,
        match="num_batches_tracked|scalar|int64|dtype|subclass|device",
    ):
        parse_model(model, input_shape=(1, 2, 2))


def test_rejects_missing_num_batches_tracked_when_stats_are_enabled() -> None:
    model = _basic_model()
    model[1].num_batches_tracked = None

    with pytest.raises(UnsupportedOpError, match="num_batches_tracked|missing"):
        parse_model(model, input_shape=(1, 2, 2))


def test_rejects_running_statistic_registered_as_parameter() -> None:
    model = _basic_model()
    del model[1].running_mean
    model[1].register_parameter(
        "running_mean",
        nn.Parameter(torch.zeros(1), requires_grad=False),
    )

    with pytest.raises(
        UnsupportedOpError,
        match="running_mean|buffer|registry|parameter schema",
    ):
        parse_model(model, input_shape=(1, 2, 2))


def test_unsafe_flatten_that_consumes_source_batch_is_rejected() -> None:
    model = nn.Sequential(
        nn.Conv2d(1, 2, 1),
        nn.BatchNorm2d(2),
        nn.Flatten(start_dim=0, end_dim=-1),
        nn.Linear(8, 2),
    ).eval()

    with pytest.raises(UnsupportedOpError, match="Flatten|batch|singleton"):
        parse_model(model, input_shape=(1, 2, 2))


class _RepeatedBatchNormOutputUse(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.bn(self.conv(inputs))
        return normalized + normalized


def test_repeated_edge_to_same_user_is_explicit_batchnorm_fanout() -> None:
    model = _RepeatedBatchNormOutputUse().eval()

    with pytest.raises(UnsupportedOpError, match="BatchNorm2d.*fan-out|fan-out.*BatchNorm2d"):
        parse_model(model, input_shape=(1, 2, 2))


class _NearTieConvBn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, bias=True)
        self.bn = nn.BatchNorm2d(2, eps=1.5484408771953614e-06)
        weights = torch.tensor(
            [
                [
                    [
                        [-17.001914978027344, -12.001204490661621, -81.09501647949219],
                        [71.63404083251953, -70.51910400390625, 24.922298431396484],
                        [-50.0478630065918, -84.30535888671875, -6.370091438293457],
                    ]
                ],
                [
                    [
                        [33.181846618652344, -91.7475357055664, 7.229316234588623],
                        [-37.597381591796875, -14.245260238647461, -44.593143463134766],
                        [-12.458145141601562, 89.95703125, -96.13422393798828],
                    ]
                ],
            ],
            dtype=torch.float32,
        )
        with torch.no_grad():
            self.conv.weight.copy_(weights)
            self.conv.bias.copy_(
                torch.tensor([-47.310245513916016, 84.98338317871094])
            )
            assert self.bn.running_mean is not None
            assert self.bn.running_var is not None
            assert self.bn.weight is not None
            assert self.bn.bias is not None
            self.bn.running_mean.copy_(
                torch.tensor([-5.263245105743408, -20.775938034057617])
            )
            self.bn.running_var.copy_(
                torch.tensor([66.8125228881836, 28.881742477416992])
            )
            self.bn.weight.copy_(
                torch.tensor([41.06906509399414, -61.68522644042969])
            )
            self.bn.bias.copy_(torch.tensor([0.0, 812.99951171875]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(inputs))


class _FixedBoundaryOutput(nn.Module):
    def __init__(self, values: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("values", values)

    def forward(self, _inputs: torch.Tensor) -> torch.Tensor:
        return self.values


def test_boundary_a_rejects_deterministic_allclose_class_change() -> None:
    source = _FixedBoundaryOutput(
        torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    ).eval()
    fused = _FixedBoundaryOutput(
        torch.tensor([1.0, 1.0 + 5.0e-7], dtype=torch.float32)
    ).eval()

    with pytest.raises(UnsupportedOpError, match="Argmax|class_id"):
        semantics._validate_fusion_boundary(
            source_model=source,
            fused_model=fused,
            input_shape=(1,),
            float_dtype="float32",
            nn=nn,
            torch=torch,
        )


def test_near_tie_class_change_is_not_hidden_by_allclose() -> None:
    model = _NearTieConvBn().eval()
    inputs = torch.linspace(-0.75, 0.75, 9).reshape(1, 3, 3)
    with torch.no_grad():
        source = model(inputs.unsqueeze(0))[0]
        fused_conv = torch.nn.utils.fusion.fuse_conv_bn_eval(model.conv, model.bn)
        fused = fused_conv(inputs)
    if not torch.allclose(source, fused, rtol=1e-5, atol=1e-6):
        pytest.skip("near-tie fixture is not allclose on this PyTorch runtime")
    source_class = int(torch.argmax(source))
    fused_class = int(torch.argmax(fused))
    if source_class == fused_class:
        pytest.skip("near-tie fixture preserves class on this PyTorch runtime")

    with pytest.raises(UnsupportedOpError, match="semantic|class|argmax"):
        parse_model(model, input_shape=(1, 3, 3))


def test_shared_relu_without_batchnorm_remains_supported() -> None:
    class SharedRelu(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.relu = nn.ReLU()

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.relu(self.relu(inputs))

    model = SharedRelu().eval()
    graph = parse_model(model, input_shape=(2, 2))
    result = infer_float_graph(
        graph,
        np.asarray([[-1.0, 0.5], [0.25, -0.75]], dtype=np.float32),
    )

    assert [type(op) for op in graph.ops] == [ReluIR, ReluIR]
    assert "input_adapter" not in graph.metadata
    np.testing.assert_array_equal(
        result.output,
        np.asarray([[0.0, 0.5], [0.25, 0.0]], dtype=np.float32),
    )
