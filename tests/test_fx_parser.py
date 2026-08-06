from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from torch2rtl.frontend.pytorch_fx import UnsupportedOpError, parse_model
from torch2rtl.ir.ops import ArgmaxIR, Conv2dIR, FlattenIR, LinearIR, ReluIR
from torch2rtl.quant.fixed_point import (
    FixedPointConfig,
    dequantize_array,
    quantize_array,
)
from torch2rtl.quant.reference import (
    QuantizedArgmaxIR,
    infer_float_graph,
    infer_quantized,
    quantize_graph,
)


def test_fx_parser_extracts_linear_relu_linear() -> None:
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 4),
    )
    graph = parse_model(model, input_shape=(16,))
    assert [type(op) for op in graph.ops] == [LinearIR, ReluIR, LinearIR]
    assert graph.linear_ops[0].in_features == 16
    assert graph.linear_ops[1].out_features == 4
    assert graph.output.shape == (4,)
    assert isinstance(quantize_graph(graph, FixedPointConfig()).ops[-1], QuantizedArgmaxIR)


def test_fx_parser_extracts_conv2d_relu_flatten_linear() -> None:
    model = nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(start_dim=0),
        nn.Linear(8, 3),
    )
    graph = parse_model(model, input_shape=(1, 4, 4))

    assert [type(op) for op in graph.ops] == [
        Conv2dIR,
        ReluIR,
        FlattenIR,
        LinearIR,
    ]
    conv = graph.ops[0]
    assert isinstance(conv, Conv2dIR)
    assert conv.in_channels == 1
    assert conv.out_channels == 2
    assert conv.output.shape == (2, 2, 2)
    assert conv.stride == (2, 2)
    assert conv.padding == (1, 1)
    assert graph.linear_ops[0].in_features == 8


def test_fx_parser_rejects_conv2d_batch_dimension() -> None:
    model = nn.Sequential(nn.Conv2d(1, 2, kernel_size=3))

    with pytest.raises(UnsupportedOpError, match="batch dimension"):
        parse_model(model, input_shape=(1, 1, 4, 4))


def test_fx_parser_rejects_grouped_conv2d() -> None:
    model = nn.Sequential(nn.Conv2d(2, 2, kernel_size=3, groups=2))

    with pytest.raises(UnsupportedOpError, match="groups=1"):
        parse_model(model, input_shape=(2, 4, 4))


def test_fx_parser_rejects_dilated_conv2d() -> None:
    model = nn.Sequential(nn.Conv2d(1, 2, kernel_size=3, dilation=2))

    with pytest.raises(UnsupportedOpError, match="dilation=1"):
        parse_model(model, input_shape=(1, 6, 6))


def test_fx_parser_rejects_model_returning_non_final_node() -> None:
    class ReturnsIntermediate(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(2, 2)
            self.second = nn.Linear(2, 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            intermediate = self.first(inputs)
            self.second(intermediate)
            return intermediate

    with pytest.raises(UnsupportedOpError, match="output must be the final node"):
        parse_model(ReturnsIntermediate(), input_shape=(2,))


def test_fx_parser_rejects_non_sequential_dependency() -> None:
    class SkipsPreviousNode(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(2, 2)
            self.relu = nn.ReLU()

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            self.first(inputs)
            return self.relu(inputs)

    with pytest.raises(UnsupportedOpError, match="single sequential dependency"):
        parse_model(SkipsPreviousNode(), input_shape=(2,))


def test_fx_parser_rejects_multiple_tensor_inputs() -> None:
    class AddsInputs(nn.Module):
        def forward(
            self,
            first: torch.Tensor,
            second: torch.Tensor,
        ) -> torch.Tensor:
            return first + second

    with pytest.raises(UnsupportedOpError, match="exactly one tensor input"):
        parse_model(AddsInputs(), input_shape=(2,))


def test_fx_parser_rejects_multiple_outputs() -> None:
    class ReturnsTuple(nn.Module):
        def forward(
            self,
            inputs: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return inputs, inputs

    with pytest.raises(UnsupportedOpError, match="tuples, lists, dictionaries"):
        parse_model(ReturnsTuple(), input_shape=(2,))


@pytest.mark.parametrize(
    "use_method",
    [False, True],
)
def test_fx_parser_rejects_parameterized_argmax(use_method: bool) -> None:
    if use_method:
        class ParameterizedArgmax(nn.Module):
            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return inputs.argmax(dim=1, keepdim=True)
    else:
        class ParameterizedArgmax(nn.Module):
            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return torch.argmax(inputs, dim=1, keepdim=True)

    with pytest.raises(UnsupportedOpError, match="only global argmax"):
        parse_model(ParameterizedArgmax(), input_shape=(2, 3))


def test_fx_parser_preserves_explicit_global_argmax() -> None:
    class GlobalArgmax(nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return torch.argmax(inputs)

    graph = parse_model(GlobalArgmax(), input_shape=(2, 3))
    assert [type(op) for op in graph.ops] == [ArgmaxIR]
    assert graph.output.shape == ()
    inputs = torch.tensor([[1.0, 9.0, 2.0], [8.0, 3.0, 4.0]])
    assert infer_float_graph(graph, inputs.numpy()).class_id == int(
        GlobalArgmax()(inputs)
    )


def test_fx_parser_rejects_linear_prefix_dimensions_before_reference() -> None:
    model = nn.Sequential(nn.Linear(3, 2))

    with pytest.raises(UnsupportedOpError, match="only a 1-D feature vector"):
        parse_model(model, input_shape=(2, 3))


def test_fx_parser_rejects_partial_flatten() -> None:
    model = nn.Sequential(nn.Flatten(start_dim=1))

    with pytest.raises(UnsupportedOpError, match="complete tensor"):
        parse_model(model, input_shape=(2, 3))


@pytest.mark.parametrize("input_shape", [(0, 3), (-1,), (True, 3)])
def test_fx_parser_rejects_invalid_static_input_shape(
    input_shape: tuple[int, ...],
) -> None:
    with pytest.raises(UnsupportedOpError, match="positive integers"):
        parse_model(nn.Sequential(nn.ReLU()), input_shape=input_shape)


def test_pytorch_float_matches_graph_ir_float() -> None:
    torch.manual_seed(7)
    model = nn.Sequential(
        nn.Linear(4, 3),
        nn.ReLU(),
        nn.Linear(3, 2),
    ).eval()
    inputs = torch.tensor([0.25, -0.5, 0.75, 1.0])
    graph = parse_model(model, input_shape=(4,))

    torch_output = model(inputs).detach().numpy()
    graph_output = infer_float_graph(graph, inputs.numpy()).logits

    assert graph.output.shape == tuple(torch_output.shape)
    assert graph_output.shape == torch_output.shape
    assert graph_output == pytest.approx(torch_output, abs=1e-6)


def test_pytorch_float_matches_graph_ir_at_each_layer() -> None:
    torch.manual_seed(11)
    model = nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=2, stride=1, padding=0),
        nn.ReLU(),
        nn.Flatten(start_dim=0),
        nn.Linear(8, 3),
    ).eval()
    inputs = torch.tensor(
        [[[0.25, -0.5, 0.75], [1.0, -0.25, 0.5], [-0.75, 0.125, 0.625]]]
    )
    graph = parse_model(model, input_shape=(1, 3, 3))

    pytorch_activations: list[torch.Tensor] = []
    current = inputs
    for layer in model:
        current = layer(current)
        pytorch_activations.append(current.detach())

    graph_result = infer_float_graph(graph, inputs.numpy())

    assert [activation.name for activation in graph_result.activations] == [
        "_0",
        "_1",
        "_2",
        "_3",
    ]
    for pytorch_value, graph_activation in zip(
        pytorch_activations,
        graph_result.activations,
        strict=True,
    ):
        assert graph_activation.values.shape == tuple(pytorch_value.shape)
        assert graph_activation.values == pytest.approx(
            pytorch_value.numpy(),
            abs=1e-6,
        )
    assert graph_result.class_id == int(torch.argmax(current))


def test_graph_ir_float_matches_exact_fixed_point_reference() -> None:
    model = nn.Sequential(nn.Linear(2, 2))
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(2))
        model[0].bias.zero_()
    cfg = FixedPointConfig(bits=8, frac_bits=6, acc_bits=16)
    graph = parse_model(model, input_shape=(2,))
    qgraph = quantize_graph(graph, cfg)
    float_input = np.asarray([1.0, 0.5])
    fixed_input = quantize_array(float_input, cfg)

    floating = infer_float_graph(graph, float_input)
    fixed = infer_quantized(qgraph, fixed_input)

    np.testing.assert_array_equal(
        dequantize_array(fixed.logits, cfg),
        floating.logits,
    )
    np.testing.assert_array_equal(
        dequantize_array(fixed.activations[0].values, cfg),
        floating.activations[0].values,
    )
