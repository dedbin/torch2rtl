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
    "mode_case",
    ["root", "conv", "batchnorm", "mixed"],
)
def test_rejects_training_state_without_mutating_source(mode_case: str) -> None:
    if mode_case == "mixed":
        model = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
        ).eval()
        model[3].train()
    else:
        model = _basic_model()
        if mode_case == "root":
            model.train()
            model[0].eval()
            model[1].eval()
        elif mode_case == "conv":
            model[0].train()
        else:
            model[1].train()
    before = _source_snapshot(model)

    with pytest.raises(UnsupportedOpError, match="eval|training"):
        parse_model(model, input_shape=(1, 2, 2))

    assert _source_snapshot(model) == before


def test_successful_fusion_preserves_source_identities_values_flags_and_hooks() -> None:
    model = _basic_model(out_channels=2, conv_bias=False)
    model.marker = ("source", 17)
    before = _source_snapshot(model)

    graph = parse_model(model, input_shape=(1, 2, 2))

    assert [type(op) for op in graph.ops] == [Conv2dIR]
    assert _source_snapshot(model) == before


def test_failed_second_pair_preserves_source_after_first_pair_was_eligible() -> None:
    model = nn.Sequential(
        nn.Conv2d(1, 1, 1),
        nn.BatchNorm2d(1),
        nn.Conv2d(1, 1, 1),
        nn.BatchNorm2d(1),
    ).eval()
    assert model[3].running_var is not None
    model[3].running_var.fill_(-1.0)
    before = _source_snapshot(model)

    with pytest.raises(UnsupportedOpError, match="running_var|variance"):
        parse_model(model, input_shape=(1, 2, 2))

    assert _source_snapshot(model) == before


class _StatefulConvBn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.counter = 0
        self.conv = nn.Conv2d(1, 1, 1)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.counter += 1
        return self.bn(self.conv(inputs))


def test_trace_time_python_state_mutation_is_rejected_without_touching_source() -> None:
    model = _StatefulConvBn().eval()
    before = _source_snapshot(model)

    with pytest.raises(UnsupportedOpError, match="state|bytecode"):
        parse_model(model, input_shape=(1, 2, 2))

    assert model.counter == 0
    assert _source_snapshot(model) == before


class _ConvFanout(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        conv_output = self.conv(inputs)
        self.bn(conv_output)
        return self.relu(conv_output)


class _BatchNormFanout(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.first = nn.ReLU()
        self.second = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.bn(self.conv(inputs))
        self.first(normalized)
        return self.second(normalized)


@pytest.mark.parametrize(
    "model",
    [_ConvFanout().eval(), _BatchNormFanout().eval()],
    ids=["conv-fanout", "batchnorm-fanout"],
)
def test_rejects_fanout_around_fusion_pair(model: nn.Module) -> None:
    with pytest.raises(UnsupportedOpError, match="fan-out|user|sequential"):
        parse_model(model, input_shape=(1, 2, 2))


@pytest.mark.parametrize(
    "model",
    [
        nn.Sequential(nn.BatchNorm2d(1)).eval(),
        nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(1),
        ).eval(),
    ],
    ids=["standalone", "not-immediately-after-conv"],
)
def test_rejects_unfusable_batchnorm(model: nn.Module) -> None:
    with pytest.raises(UnsupportedOpError, match="BatchNorm2d|batch.?norm|fusion"):
        parse_model(model, input_shape=(1, 2, 2))


def test_rejects_public_nchw_even_when_pair_is_fusable() -> None:
    with pytest.raises(UnsupportedOpError, match="batch dimension|NCHW|CHW"):
        parse_model(_basic_model(), input_shape=(1, 1, 2, 2))


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
        fx_frontend._validate_fusion_boundary(
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
