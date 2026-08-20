from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

import torch2rtl.frontend.pytorch_fx as pytorch_fx
from torch2rtl.frontend.pytorch_fx import UnsupportedOpError, parse_model


def _public_state(model: nn.Module) -> tuple[object, ...]:
    return tuple(
        (
            name,
            type(module),
            module.training,
            tuple(
                (tensor_name, tensor.detach().cpu().numpy().copy())
                for tensor_name, tensor in module.named_parameters(recurse=False)
            ),
            tuple(
                (tensor_name, tensor.detach().cpu().numpy().copy())
                for tensor_name, tensor in module.named_buffers(recurse=False)
                if tensor is not None
            ),
        )
        for name, module in model.named_modules(remove_duplicate=False)
    )


def _assert_public_state_equal(
    before: tuple[object, ...],
    after: tuple[object, ...],
) -> None:
    assert len(before) == len(after)
    for left, right in zip(before, after, strict=True):
        assert left[:3] == right[:3]
        for left_group, right_group in zip(left[3:], right[3:], strict=True):
            assert [item[0] for item in left_group] == [item[0] for item in right_group]
            for (_, left_values), (_, right_values) in zip(
                left_group, right_group, strict=True
            ):
                np.testing.assert_array_equal(left_values, right_values)


def test_invalid_input_shape_precedes_hostile_model_validation_and_execution() -> None:
    calls = {"forward": 0}

    class Hostile(nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            calls["forward"] += 1
            raise AssertionError("must not run")

    with pytest.raises(
        UnsupportedOpError,
        match=(
            "^Unsupported input shape: positive integers are required; "
            "dimension at index 0 is not a positive built-in int$"
        ),
    ):
        parse_model(Hostile(), input_shape=(0,))
    assert calls == {"forward": 0}


def test_invalid_fusion_pair_precedes_unsupported_downstream_node() -> None:
    model = nn.Sequential(
        nn.Conv2d(1, 1, 1),
        nn.BatchNorm2d(1),
        nn.Identity(),
    ).eval()
    assert model[1].running_var is not None
    model[1].running_var.fill_(-1.0)
    before = _public_state(model)

    with pytest.raises(
        UnsupportedOpError,
        match="^Unsupported BatchNorm2d running_var: values must be non-negative$",
    ):
        parse_model(model, input_shape=(1, 2, 2))
    _assert_public_state_equal(before, _public_state(model))


def test_trace_mutation_error_precedes_trace_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mutating_trace(trace_model: nn.Module) -> object:
        trace_model.training = True
        raise LookupError("external trace failure")

    monkeypatch.setattr(torch.fx, "symbolic_trace", mutating_trace)
    model = nn.Sequential(nn.ReLU()).eval()
    with pytest.raises(
        UnsupportedOpError,
        match="^Unsupported Python state mutation during FX symbolic trace$",
    ) as captured:
        pytorch_fx._parse_model_impl(model, (2,), None, torch, nn)
    assert captured.value.__cause__ is None
    assert model.training is False


def test_external_trace_error_has_stable_prefix_and_exact_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_trace(_trace_model: nn.Module) -> object:
        raise LookupError("external trace failure")

    monkeypatch.setattr(torch.fx, "symbolic_trace", failing_trace)
    with pytest.raises(
        UnsupportedOpError,
        match=(
            "^Unsupported FX symbolic trace semantics: external trace failure$"
        ),
    ) as captured:
        pytorch_fx._parse_model_impl(
            nn.Sequential(nn.ReLU()).eval(),
            (2,),
            None,
            torch,
            nn,
        )
    assert type(captured.value.__cause__) is LookupError
    assert captured.value.__cause__.args == ("external trace failure",)


def test_integrity_hardening_precedes_patched_semantic_boundaries_on_baseline_b(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fail_a(**_kwargs: object) -> None:
        calls.append("A")
        raise UnsupportedOpError("characterization boundary A")

    def fail_b(**_kwargs: object) -> None:
        calls.append("B")
        raise UnsupportedOpError("characterization boundary B")

    monkeypatch.setattr(
        pytorch_fx._pipeline_component,
        "_validate_fusion_boundary",
        fail_a,
    )
    monkeypatch.setattr(
        pytorch_fx._pipeline_component,
        "_validate_lowered_semantics",
        fail_b,
    )
    model = nn.Sequential(nn.Conv2d(1, 1, 1), nn.BatchNorm2d(1)).eval()

    with pytest.raises(
        UnsupportedOpError,
        match=(
            "^Unsupported modified frontend component pipeline$"
        ),
    ):
        parse_model(model, input_shape=(1, 2, 2))
    assert calls == []


def test_argmax_terminal_error_precedes_generic_unsupported_node() -> None:
    class OperationAfterArgmax(nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs.argmax().reshape(())

    with pytest.raises(
        UnsupportedOpError,
        match="^Unsupported FX graph: argmax must be the final operation$",
    ):
        parse_model(OperationAfterArgmax(), input_shape=(2, 3))


@pytest.mark.parametrize("batchnorm", [nn.BatchNorm1d(2), nn.BatchNorm3d(2)])
def test_non_2d_batchnorm_rejection_is_characterized(batchnorm: nn.Module) -> None:
    with pytest.raises(
        UnsupportedOpError,
        match=(
            "^Unsupported dtype torch.int64 for tensor 0.num_batches_tracked; "
            "only float32 and float64 are supported$"
        ),
    ):
        parse_model(nn.Sequential(batchnorm).eval(), input_shape=(2, 2, 2))


@pytest.mark.parametrize(
    ("input_dtype", "expected"),
    [
        (torch.float32, "float32"),
        (np.dtype(np.float32), "float32"),
        ("float32", "float32"),
        ("torch.float32", "float32"),
        (torch.float64, "float64"),
        (np.dtype(np.float64), "float64"),
        ("float64", "float64"),
        ("torch.float64", "float64"),
    ],
)
def test_accepted_input_dtype_forms_are_characterized(
    input_dtype: object,
    expected: str,
) -> None:
    graph = parse_model(nn.Sequential(nn.ReLU()), (2,), input_dtype=input_dtype)
    assert graph.input.dtype == expected
    assert graph.output.dtype == expected


def test_non_batchnorm_success_does_not_mutate_source() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU()).eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[0.25, -0.5], [1.0, 0.75]]))
        model[0].bias.copy_(torch.tensor([0.125, -0.25]))
    before = copy.deepcopy(_public_state(model))

    parse_model(model, input_shape=(2,))

    _assert_public_state_equal(before, _public_state(model))
