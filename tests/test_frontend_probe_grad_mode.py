from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import torch2rtl.frontend.pytorch_fx as frontend
from torch2rtl.frontend import _semantics as semantics
from torch2rtl.ir.graph import GraphIR
from torch2rtl.ir.tensor import TensorIR


_RECORDED_GRAD_MODES: list[bool] = []


class _GradRecordingIdentity(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        _RECORDED_GRAD_MODES.append(torch.is_grad_enabled())
        return value


def _identity_graph() -> GraphIR:
    tensor = TensorIR(name="input", shape=(2,), dtype="float32")
    return GraphIR(input=tensor, output=tensor, ops=())


def test_boundary_a_concrete_probe_preserves_no_grad_semantics() -> None:
    _RECORDED_GRAD_MODES.clear()
    assert torch.is_grad_enabled() is True
    result = semantics._run_concrete_probe(
        _GradRecordingIdentity().eval(),
        torch.tensor([0.25, -0.5], dtype=torch.float32),
        "grad-mode characterization",
        torch.nn,
        torch,
        copy_model=False,
        security_context=frontend._TRUSTED_CALLBACK_SECURITY_CONTEXT,
        checked_call=frontend._checked_untrusted_call,
    )
    assert np.array_equal(result.detach().numpy(), np.array([0.25, -0.5]))
    assert _RECORDED_GRAD_MODES == [False]
    assert torch.is_grad_enabled() is True


def test_boundary_b_lowered_probe_preserves_no_grad_semantics() -> None:
    _RECORDED_GRAD_MODES.clear()
    assert torch.is_grad_enabled() is True
    semantics._validate_lowered_semantics(
        _GradRecordingIdentity().eval(),
        _identity_graph(),
        (2,),
        "float32",
        torch.nn,
        torch,
        copy_model=False,
        security_context=frontend._TRUSTED_CALLBACK_SECURITY_CONTEXT,
        checked_call=frontend._checked_untrusted_call,
    )
    assert _RECORDED_GRAD_MODES == [False, False]
    assert torch.is_grad_enabled() is True


def test_probe_restores_preexisting_disabled_grad_mode() -> None:
    _RECORDED_GRAD_MODES.clear()
    with torch.no_grad():
        semantics._run_concrete_probe(
            _GradRecordingIdentity().eval(),
            torch.tensor([1.0, 2.0]),
            "disabled caller grad mode",
            torch.nn,
            torch,
            copy_model=False,
            security_context=frontend._TRUSTED_CALLBACK_SECURITY_CONTEXT,
            checked_call=frontend._checked_untrusted_call,
        )
        assert torch.is_grad_enabled() is False
    assert _RECORDED_GRAD_MODES == [False]
    assert torch.is_grad_enabled() is True
