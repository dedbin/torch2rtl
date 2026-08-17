from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from torch2rtl.backend.systemverilog.emit import emit_systemverilog
from torch2rtl.frontend.pytorch_fx import parse_model
from torch2rtl.ir.ops import ArgmaxIR, Conv2dIR, FlattenIR, LinearIR, ReluIR
from torch2rtl.quant.fixed_point import FixedPointConfig
from torch2rtl.quant.reference import infer_float_graph


_BATCHNORM_SUPPORTS_BIAS = (
    "bias" in inspect.signature(nn.BatchNorm2d).parameters
)


def _source_snapshot(model: nn.Module) -> tuple[object, ...]:
    modules = tuple(
        (name, id(module), type(module), module.training)
        for name, module in model.named_modules(remove_duplicate=False)
    )

    def tensor_state(tensor: torch.Tensor) -> tuple[object, ...]:
        values = tensor.detach().cpu().contiguous().numpy()
        return (
            id(tensor),
            type(tensor),
            tensor.dtype,
            tuple(tensor.shape),
            tensor.requires_grad,
            values.tobytes(),
        )

    parameters = tuple(
        (name, tensor_state(parameter))
        for name, parameter in model.named_parameters(remove_duplicate=False)
    )
    buffers = tuple(
        (name, tensor_state(buffer))
        for name, buffer in model.named_buffers(remove_duplicate=False)
    )
    return modules, parameters, buffers


def _manual_conv_bn_model(
    *,
    conv_bias: bool,
    bn_mode: str,
    dtype: torch.dtype,
) -> nn.Sequential:
    if bn_mode == "affine-no-bias" and not _BATCHNORM_SUPPORTS_BIAS:
        pytest.skip("installed BatchNorm2d constructor has no bias keyword")
    affine = bn_mode != "no-affine"
    batchnorm_kwargs: dict[str, object] = {}
    if bn_mode == "affine-no-bias":
        batchnorm_kwargs["bias"] = False
    conv = nn.Conv2d(1, 1, kernel_size=1, bias=conv_bias, dtype=dtype)
    bn = nn.BatchNorm2d(
        1,
        eps=1.0,
        affine=affine,
        track_running_stats=True,
        dtype=dtype,
        **batchnorm_kwargs,
    )
    with torch.no_grad():
        conv.weight.fill_(2.0)
        if conv.bias is not None:
            conv.bias.fill_(3.0)
        assert bn.running_mean is not None
        assert bn.running_var is not None
        bn.running_mean.fill_(1.0)
        bn.running_var.fill_(3.0)
        if bn.weight is not None:
            bn.weight.fill_(4.0)
        if bn.bias is not None:
            bn.bias.fill_(5.0)
    return nn.Sequential(conv, bn).eval()


def _assert_public_float_equivalence(
    model: nn.Module,
    graph: object,
    inputs: torch.Tensor,
) -> None:
    with torch.no_grad():
        source = model(inputs.unsqueeze(0))
    assert isinstance(source, torch.Tensor)
    if source.ndim > 0:
        assert source.shape[0] == 1
        source = source[0]
    result = infer_float_graph(graph, inputs.detach().cpu().numpy())  # type: ignore[arg-type]
    expected = source.detach().cpu().numpy()
    assert result.output.shape == expected.shape
    if inputs.dtype is torch.float64:
        np.testing.assert_allclose(result.output, expected, rtol=1e-12, atol=1e-12)
    else:
        np.testing.assert_allclose(result.output, expected, rtol=1e-5, atol=1e-6)
    assert result.class_id == int(np.argmax(expected))


@pytest.mark.parametrize(
    ("conv_bias", "bn_mode", "expected_weight", "expected_bias"),
    [
        pytest.param(True, "affine-bias", 4.0, 9.0, id="conv-bias_bn-bias"),
        pytest.param(False, "affine-bias", 4.0, 3.0, id="no-conv-bias_bn-bias"),
        pytest.param(
            True,
            "affine-no-bias",
            4.0,
            4.0,
            id="conv-bias_no-bn-bias",
        ),
        pytest.param(
            False,
            "affine-no-bias",
            4.0,
            -2.0,
            id="no-conv-bias_no-bn-bias",
        ),
        pytest.param(True, "no-affine", 1.0, 1.0, id="conv-bias_no-bn-affine"),
        pytest.param(
            False,
            "no-affine",
            1.0,
            -0.5,
            id="no-conv-bias_no-bn-affine",
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fuses_conv_bn_bias_affine_and_dtype_matrix(
    conv_bias: bool,
    bn_mode: str,
    expected_weight: float,
    expected_bias: float,
    dtype: torch.dtype,
) -> None:
    model = _manual_conv_bn_model(
        conv_bias=conv_bias,
        bn_mode=bn_mode,
        dtype=dtype,
    )
    before = _source_snapshot(model)
    official = torch.nn.utils.fusion.fuse_conv_bn_eval(model[0], model[1])
    assert _source_snapshot(model) == before

    graph = parse_model(
        model,
        input_shape=(1, 2, 2),
        input_dtype=dtype,
    )
    assert _source_snapshot(model) == before

    assert [type(op) for op in graph.ops] == [Conv2dIR]
    conv = graph.ops[0]
    assert isinstance(conv, Conv2dIR)
    expected_numpy_dtype = np.dtype(np.float64 if dtype is torch.float64 else np.float32)
    assert conv.weight.dtype == expected_numpy_dtype
    assert conv.bias is not None
    assert conv.bias.dtype == expected_numpy_dtype
    np.testing.assert_array_equal(
        conv.weight,
        np.full((1, 1, 1, 1), expected_weight, dtype=expected_numpy_dtype),
    )
    np.testing.assert_array_equal(
        conv.bias,
        np.asarray([expected_bias], dtype=expected_numpy_dtype),
    )
    np.testing.assert_array_equal(
        conv.weight,
        official.weight.detach().cpu().numpy(),
    )
    assert official.bias is not None
    np.testing.assert_array_equal(
        conv.bias,
        official.bias.detach().cpu().numpy(),
    )
    assert graph.output.shape == (1, 2, 2)
    assert graph.metadata["input_adapter"] == {"kind": "singleton_batch_n1"}
    transformations = graph.metadata["transformations"]
    assert isinstance(transformations, list)
    assert len(transformations) == 1
    record = transformations[0]
    assert record["kind"] == "conv2d_batchnorm2d_fusion"
    assert record["conv_target"] == "0"
    assert record["batchnorm_target"] == "1"
    assert all(type(record[name]) is str and record[name] for name in (
        "conv_node",
        "conv_target",
        "batchnorm_node",
        "batchnorm_target",
        "fused_target",
    ))

    inputs = torch.tensor(
        [[[-1.0, 0.25], [0.75, 1.0]]],
        dtype=dtype,
    )
    with torch.no_grad():
        source = model(inputs.unsqueeze(0))[0]
        fused = official(inputs)
    if dtype is torch.float64:
        np.testing.assert_allclose(
            source.detach().cpu().numpy(),
            fused.detach().cpu().numpy(),
            rtol=1e-12,
            atol=1e-12,
        )
    else:
        np.testing.assert_allclose(
            source.detach().cpu().numpy(),
            fused.detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-6,
        )
    assert int(torch.argmax(source)) == int(torch.argmax(fused))
    _assert_public_float_equivalence(model, graph, inputs)


def test_two_pairs_match_independent_manual_formulas_and_metadata() -> None:
    dtype = torch.float64
    first_conv = nn.Conv2d(1, 2, 1, bias=False, dtype=dtype)
    first_bn_kwargs: dict[str, object] = {}
    if _BATCHNORM_SUPPORTS_BIAS:
        first_bn_kwargs["bias"] = False
    first_bn = nn.BatchNorm2d(
        2,
        eps=0.25,
        affine=True,
        dtype=dtype,
        **first_bn_kwargs,
    )
    second_conv = nn.Conv2d(2, 1, 1, bias=True, dtype=dtype)
    second_bn = nn.BatchNorm2d(1, eps=0.25, affine=True, dtype=dtype)
    model = nn.Sequential(
        first_conv,
        first_bn,
        nn.ReLU(),
        second_conv,
        second_bn,
    ).eval()
    with torch.no_grad():
        first_conv.weight.copy_(torch.tensor([2.0, -3.0], dtype=dtype).reshape(2, 1, 1, 1))
        first_bn.running_mean.copy_(torch.tensor([1.0, -2.0], dtype=dtype))
        first_bn.running_var.copy_(torch.tensor([0.75, 3.75], dtype=dtype))
        assert first_bn.weight is not None
        first_bn.weight.copy_(torch.tensor([4.0, -2.0], dtype=dtype))
        if _BATCHNORM_SUPPORTS_BIAS:
            assert first_bn.bias is None
            first_beta = np.zeros(2, dtype=np.float64)
        else:
            assert first_bn.bias is not None
            first_bn.bias.copy_(torch.tensor([0.75, -1.25], dtype=dtype))
            first_beta = np.asarray([0.75, -1.25], dtype=np.float64)

        second_conv.weight.copy_(
            torch.tensor([0.5, -4.0], dtype=dtype).reshape(1, 2, 1, 1)
        )
        assert second_conv.bias is not None
        second_conv.bias.fill_(2.0)
        second_bn.running_mean.fill_(-1.0)
        second_bn.running_var.fill_(3.75)
        assert second_bn.weight is not None
        assert second_bn.bias is not None
        second_bn.weight.fill_(-4.0)
        second_bn.bias.fill_(0.5)

    before = _source_snapshot(model)
    official_first = torch.nn.utils.fusion.fuse_conv_bn_eval(first_conv, first_bn)
    official_second = torch.nn.utils.fusion.fuse_conv_bn_eval(second_conv, second_bn)
    assert _source_snapshot(model) == before

    graph = parse_model(model, input_shape=(1, 2, 3), input_dtype=dtype)

    assert _source_snapshot(model) == before
    assert [type(op) for op in graph.ops] == [Conv2dIR, ReluIR, Conv2dIR]
    fused_convs = [op for op in graph.ops if isinstance(op, Conv2dIR)]
    assert len(fused_convs) == 2
    expected_weights = (
        np.asarray([8.0, 3.0], dtype=np.float64).reshape(2, 1, 1, 1),
        np.asarray([-1.0, 8.0], dtype=np.float64).reshape(1, 2, 1, 1),
    )
    expected_biases = (
        np.asarray([-4.0, -2.0], dtype=np.float64) + first_beta,
        np.asarray([-5.5], dtype=np.float64),
    )
    for fused_conv, official, expected_weight, expected_bias in zip(
        fused_convs,
        (official_first, official_second),
        expected_weights,
        expected_biases,
        strict=True,
    ):
        np.testing.assert_array_equal(fused_conv.weight, expected_weight)
        np.testing.assert_array_equal(fused_conv.bias, expected_bias)
        np.testing.assert_array_equal(
            fused_conv.weight,
            official.weight.detach().cpu().numpy(),
        )
        assert official.bias is not None
        np.testing.assert_array_equal(
            fused_conv.bias,
            official.bias.detach().cpu().numpy(),
        )

    transformations = graph.metadata["transformations"]
    assert [record["conv_target"] for record in transformations] == ["0", "3"]
    assert [record["batchnorm_target"] for record in transformations] == ["1", "4"]
    assert len({record["conv_node"] for record in transformations}) == 2
    assert len({record["batchnorm_node"] for record in transformations}) == 2
    assert len({record["fused_target"] for record in transformations}) == 2

    inputs = torch.tensor(
        [[[-0.75, 0.25, 1.0], [0.5, -0.125, 0.875]]],
        dtype=dtype,
    )
    official_model = nn.Sequential(
        official_first,
        nn.ReLU(),
        official_second,
    ).eval()
    with torch.no_grad():
        source = model(inputs.unsqueeze(0))[0]
        official_output = official_model(inputs)
    np.testing.assert_allclose(
        source.detach().cpu().numpy(),
        official_output.detach().cpu().numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    assert int(torch.argmax(source)) == int(torch.argmax(official_output))
    _assert_public_float_equivalence(model, graph, inputs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_eps_with_positive_running_variance_is_supported(
    dtype: torch.dtype,
) -> None:
    model = _manual_conv_bn_model(
        conv_bias=True,
        bn_mode="affine-bias",
        dtype=dtype,
    )
    model[1].eps = 0.0
    assert model[1].running_var is not None
    model[1].running_var.fill_(4.0)
    before = _source_snapshot(model)
    official = torch.nn.utils.fusion.fuse_conv_bn_eval(model[0], model[1])

    graph = parse_model(model, input_shape=(1, 2, 2), input_dtype=dtype)

    assert _source_snapshot(model) == before
    fused_conv = graph.ops[0]
    assert isinstance(fused_conv, Conv2dIR)
    expected_dtype = np.dtype(np.float64 if dtype is torch.float64 else np.float32)
    np.testing.assert_array_equal(
        fused_conv.weight,
        np.asarray([4.0], dtype=expected_dtype).reshape(1, 1, 1, 1),
    )
    np.testing.assert_array_equal(
        fused_conv.bias,
        np.asarray([9.0], dtype=expected_dtype),
    )
    np.testing.assert_array_equal(
        fused_conv.weight,
        official.weight.detach().cpu().numpy(),
    )
    assert official.bias is not None
    np.testing.assert_array_equal(
        fused_conv.bias,
        official.bias.detach().cpu().numpy(),
    )
    inputs = torch.tensor(
        [[[-1.0, 0.25], [0.75, 1.0]]],
        dtype=dtype,
    )
    _assert_public_float_equivalence(model, graph, inputs)


def test_terminal_bn_removes_only_the_singleton_batch_dimension() -> None:
    model = nn.Sequential(
        nn.Conv2d(1, 1, kernel_size=1),
        nn.BatchNorm2d(1),
    ).eval()

    graph = parse_model(model, input_shape=(1, 2, 3))

    assert graph.output.shape == (1, 2, 3)
    inputs = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3) / 4
    _assert_public_float_equivalence(model, graph, inputs)


def test_conv_bn_relu_and_multiple_pairs_preserve_public_semantics() -> None:
    torch.manual_seed(20260817)
    model = nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=1, bias=False),
        nn.BatchNorm2d(2),
        nn.ReLU(),
        nn.Conv2d(2, 1, kernel_size=1, bias=True),
        nn.BatchNorm2d(1, affine=False),
    ).eval()
    with torch.no_grad():
        model[1].running_mean.copy_(torch.tensor([0.25, -0.5]))
        model[1].running_var.copy_(torch.tensor([0.5, 2.0]))
        model[4].running_mean.fill_(-0.125)
        model[4].running_var.fill_(0.75)

    graph = parse_model(model, input_shape=(1, 2, 3))

    assert [type(op) for op in graph.ops] == [Conv2dIR, ReluIR, Conv2dIR]
    transformations = graph.metadata["transformations"]
    assert [record["conv_target"] for record in transformations] == ["0", "3"]
    assert [record["batchnorm_target"] for record in transformations] == ["1", "4"]
    assert len({record["fused_target"] for record in transformations}) == 2
    inputs = torch.tensor(
        [[[-0.75, 0.25, 1.0], [0.5, -0.125, 0.875]]],
        dtype=torch.float32,
    )
    _assert_public_float_equivalence(model, graph, inputs)


class _ReusedConvBnRelu(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        current = self.bn(self.conv(inputs))
        current = self.relu(current)
        current = self.bn(self.conv(current))
        return self.relu(current)


def test_reused_conv_bn_and_relu_create_distinct_fused_call_sites() -> None:
    model = _ReusedConvBnRelu().eval()
    with torch.no_grad():
        model.conv.weight.fill_(0.75)
        model.conv.bias.fill_(-0.125)
        model.bn.weight.fill_(-1.25)
        model.bn.bias.fill_(0.5)
        model.bn.running_mean.fill_(0.25)
        model.bn.running_var.fill_(1.5)

    graph = parse_model(model, input_shape=(1, 2, 2))

    assert [type(op) for op in graph.ops] == [
        Conv2dIR,
        ReluIR,
        Conv2dIR,
        ReluIR,
    ]
    transformations = graph.metadata["transformations"]
    assert [record["conv_target"] for record in transformations] == ["conv", "conv"]
    assert [record["batchnorm_target"] for record in transformations] == ["bn", "bn"]
    assert len({record["fused_target"] for record in transformations}) == 2
    assert [record["conv_node"] for record in transformations] == ["conv", "conv_1"]
    assert [record["batchnorm_node"] for record in transformations] == ["bn", "bn_1"]
    inputs = torch.tensor([[[0.5, -0.25], [1.0, -0.75]]])
    _assert_public_float_equivalence(model, graph, inputs)


class _PartiallyFusedSharedConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        current = self.bn(self.conv(inputs))
        return self.conv(current)


def test_shared_conv_unfused_call_remains_semantically_independent() -> None:
    model = _PartiallyFusedSharedConv().eval()
    with torch.no_grad():
        model.conv.weight.fill_(1.5)
        model.conv.bias.fill_(-0.25)
        model.bn.weight.fill_(0.75)
        model.bn.bias.fill_(0.125)
        model.bn.running_mean.fill_(-0.5)
        model.bn.running_var.fill_(2.0)

    graph = parse_model(model, input_shape=(1, 2, 2))

    assert [type(op) for op in graph.ops] == [Conv2dIR, Conv2dIR]
    assert len(graph.metadata["transformations"]) == 1
    inputs = torch.tensor([[[-0.5, 0.25], [0.75, 1.0]]])
    _assert_public_float_equivalence(model, graph, inputs)


def test_safe_batch_relative_flatten_is_accepted() -> None:
    model = nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=1),
        nn.BatchNorm2d(2),
        nn.Flatten(start_dim=-3, end_dim=-1),
        nn.Linear(8, 3),
    ).eval()

    graph = parse_model(model, input_shape=(1, 2, 2))

    assert [type(op) for op in graph.ops] == [Conv2dIR, FlattenIR, LinearIR]
    assert graph.output.shape == (3,)
    inputs = torch.tensor([[[0.25, -0.5], [0.75, 1.0]]])
    _assert_public_float_equivalence(model, graph, inputs)


class _ConvBnGlobalArgmax(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.bn(self.conv(inputs)))


def test_explicit_global_argmax_after_fusion_remains_a_scalar() -> None:
    model = _ConvBnGlobalArgmax().eval()
    graph = parse_model(model, input_shape=(1, 2, 2))
    inputs = torch.tensor([[[0.25, -0.5], [0.75, 1.0]]])
    result = infer_float_graph(graph, inputs.numpy())

    assert isinstance(graph.ops[-1], ArgmaxIR)
    assert graph.output.shape == ()
    assert result.output.shape == ()
    with torch.no_grad():
        expected = model(inputs.unsqueeze(0))
    assert expected.shape == ()
    assert int(result.output) == result.class_id == int(expected)


def test_transform_metadata_is_deterministic_for_repeated_parse() -> None:
    model = _ReusedConvBnRelu().eval()

    first = parse_model(model, input_shape=(1, 2, 2))
    second = parse_model(model, input_shape=(1, 2, 2))

    assert first.metadata["input_adapter"] == second.metadata["input_adapter"]
    assert first.metadata["transformations"] == second.metadata["transformations"]


class _FusedTargetNameCollision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self._torch2rtl_fused_conv_bn_0 = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        current = self.bn(self.conv(inputs))
        return self._torch2rtl_fused_conv_bn_0(current)


def test_fused_target_does_not_collide_with_user_module_name() -> None:
    model = _FusedTargetNameCollision().eval()

    graph = parse_model(model, input_shape=(1, 2, 2))

    assert [type(op) for op in graph.ops] == [Conv2dIR, ReluIR]
    record = graph.metadata["transformations"][0]
    assert record["fused_target"] != "_torch2rtl_fused_conv_bn_0"
    assert record["fused_target"].startswith("_torch2rtl_fused_conv_bn_0")


def test_fixed_seed_randomized_differential_sweep() -> None:
    modes = ["affine-bias", "no-affine"]
    if _BATCHNORM_SUPPORTS_BIAS:
        modes.insert(1, "affine-no-bias")
    conv_configs = (
        (1, 1, 0, 1),
        (2, 1, 0, 1),
        (3, 1, 1, 1),
        (2, 2, 1, 1),
    )

    def eps_for_stats(stats_case: int) -> float:
        return (0.0, 0.25, 0.0, 1.0)[stats_case]

    def new_batchnorm(
        channels: int,
        mode: str,
        eps: float,
        dtype: torch.dtype,
    ) -> nn.BatchNorm2d:
        kwargs: dict[str, object] = {}
        if mode == "affine-no-bias":
            kwargs["bias"] = False
        return nn.BatchNorm2d(
            channels,
            eps=eps,
            affine=mode != "no-affine",
            dtype=dtype,
            **kwargs,
        )

    def initialize_pair(
        conv: nn.Conv2d,
        batchnorm: nn.BatchNorm2d,
        stats_case: int,
    ) -> None:
        with torch.no_grad():
            conv.weight.uniform_(-0.8, 0.8)
            if conv.bias is not None:
                conv.bias.uniform_(-0.3, 0.3)
            assert batchnorm.running_mean is not None
            assert batchnorm.running_var is not None
            batchnorm.running_mean.uniform_(-0.5, 0.5)
            batchnorm.running_var.uniform_(0.25, 2.0)
            if stats_case == 1:
                batchnorm.running_var[0] = 0.0
            elif stats_case == 2:
                batchnorm.running_var[0] = torch.finfo(conv.weight.dtype).tiny
            elif stats_case == 3:
                batchnorm.running_mean[0] = 1.0e3
                batchnorm.running_var[0] = 1.0e6
            if batchnorm.weight is not None:
                batchnorm.weight.uniform_(0.5, 1.5)
            if batchnorm.bias is not None:
                batchnorm.bias.uniform_(-0.5, 0.5)

    axis_product = tuple(
        (dtype, pair_count, first_conv_bias)
        for dtype in (torch.float32, torch.float64)
        for pair_count in (1, 2)
        for first_conv_bias in (False, True)
    )
    for variant_index in range(4):
        for axis_index, (
            dtype,
            pair_count,
            first_conv_bias,
        ) in enumerate(axis_product):
            case_index = variant_index * len(axis_product) + axis_index
            torch.manual_seed(2026081800 + case_index)
            mode = modes[(axis_index + variant_index) % len(modes)]
            second_mode = modes[(axis_index + variant_index + 1) % len(modes)]
            kernel, stride, padding, groups = conv_configs[
                (axis_index + variant_index) % len(conv_configs)
            ]
            stats_case = variant_index
            eps = eps_for_stats(stats_case)
            input_channels = 1 + (axis_index + variant_index) % 3
            middle_channels = 1 + (axis_index + 2 * variant_index) % 3
            output_channels = 1 + (2 * axis_index + variant_index) % 3
            first_conv = nn.Conv2d(
                input_channels,
                middle_channels,
                kernel,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=first_conv_bias,
                dtype=dtype,
            )
            first_bn = new_batchnorm(middle_channels, mode, eps, dtype)
            initialize_pair(first_conv, first_bn, stats_case)

            if pair_count == 1:
                model = nn.Sequential(first_conv, first_bn).eval()
                official_convs = (
                    torch.nn.utils.fusion.fuse_conv_bn_eval(first_conv, first_bn),
                )
                official_model = nn.Sequential(*official_convs).eval()
            else:
                second_kernel = 1 if (axis_index + variant_index) % 2 else 3
                second_padding = 0 if second_kernel == 1 else 1
                second_conv = nn.Conv2d(
                    middle_channels,
                    output_channels,
                    second_kernel,
                    stride=1,
                    padding=second_padding,
                    groups=1,
                    bias=bool((axis_index + variant_index) % 2),
                    dtype=dtype,
                )
                second_stats_case = (stats_case + 1) % 4
                second_bn = new_batchnorm(
                    output_channels,
                    second_mode,
                    eps_for_stats(second_stats_case),
                    dtype,
                )
                initialize_pair(second_conv, second_bn, second_stats_case)
                model = nn.Sequential(
                    first_conv,
                    first_bn,
                    nn.ReLU(),
                    second_conv,
                    second_bn,
                ).eval()
                official_convs = (
                    torch.nn.utils.fusion.fuse_conv_bn_eval(first_conv, first_bn),
                    torch.nn.utils.fusion.fuse_conv_bn_eval(second_conv, second_bn),
                )
                official_model = nn.Sequential(
                    official_convs[0],
                    nn.ReLU(),
                    official_convs[1],
                ).eval()

            before = _source_snapshot(model)
            inputs = torch.linspace(
                -0.75,
                0.75,
                input_channels * 5 * 6,
                dtype=dtype,
            ).reshape(input_channels, 5, 6)
            with torch.no_grad():
                source = model(inputs.unsqueeze(0))[0]
                official_output = official_model(inputs)
            rtol, atol = (
                (1e-5, 1e-6) if dtype is torch.float32 else (1e-12, 1e-12)
            )
            np.testing.assert_allclose(
                source.detach().cpu().numpy(),
                official_output.detach().cpu().numpy(),
                rtol=rtol,
                atol=atol,
            )
            assert int(torch.argmax(source)) == int(torch.argmax(official_output))

            graph = parse_model(
                model,
                input_shape=tuple(inputs.shape),
                input_dtype=dtype,
            )

            assert _source_snapshot(model) == before
            fused_convs = [op for op in graph.ops if isinstance(op, Conv2dIR)]
            assert len(fused_convs) == len(official_convs)
            assert len(graph.metadata["transformations"]) == len(official_convs)
            for fused_conv, official_conv in zip(
                fused_convs,
                official_convs,
                strict=True,
            ):
                np.testing.assert_array_equal(
                    fused_conv.weight,
                    official_conv.weight.detach().cpu().numpy(),
                )
                assert official_conv.bias is not None
                np.testing.assert_array_equal(
                    fused_conv.bias,
                    official_conv.bias.detach().cpu().numpy(),
                )
            result = infer_float_graph(graph, inputs.cpu().numpy())
            np.testing.assert_allclose(
                result.output,
                source.detach().cpu().numpy(),
                rtol=rtol,
                atol=atol,
            )
            assert result.class_id == int(torch.argmax(source))


def test_model_without_batchnorm_keeps_old_graph_and_metadata_contract(
    tmp_path: Path,
) -> None:
    model = nn.Sequential(
        nn.Conv2d(1, 1, kernel_size=1),
        nn.ReLU(),
    ).eval()

    graph = parse_model(model, input_shape=(1, 2, 2))

    assert [type(op) for op in graph.ops] == [Conv2dIR, ReluIR]
    assert graph.metadata == {"source": "Sequential"}
    assert "input_adapter" not in graph.metadata
    assert "transformations" not in graph.metadata
    emit_systemverilog(
        graph,
        FixedPointConfig(),
        tmp_path,
        vector_count=1,
        seed=5,
    )
    report = json.loads(
        (tmp_path / "report.json").read_text(encoding="utf-8")
    )
    assert "input_adapter" not in report["graph"]
    assert "transformations" not in report["graph"]
