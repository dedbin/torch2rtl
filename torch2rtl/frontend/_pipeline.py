from __future__ import annotations

from typing import Sequence

from torch2rtl.frontend._errors import UnsupportedOpError
from torch2rtl.frontend._fusion import _fuse_conv_batchnorm_eval
from torch2rtl.frontend._lowering import lower_fx_graph, _validate_input_shape
from torch2rtl.frontend._model_contract import _validate_model_semantics
from torch2rtl.frontend._semantics import (
    _validate_fusion_boundary,
    _validate_lowered_semantics,
)
from torch2rtl.frontend._state_guard import (
    _copy_model_for_tracing,
    _restore_module_class_definitions,
    _snapshot_module_class_definitions,
    _snapshot_module_state,
    _trusted_named_modules,
)
from torch2rtl.ir.graph import GraphIR


def _parse_model_impl(
    model: object,
    input_shape: Sequence[int],
    input_dtype: object | None,
    torch: object,
    nn: object,
    *,
    _security_context: tuple[object, ...] | None = None,
    _checked_call: object | None = None,
) -> GraphIR:
    normalized_input_shape = _validate_input_shape(input_shape)
    float_dtype = _validate_model_semantics(model, nn, torch, input_dtype)
    trace_model = _copy_model_for_tracing(model, nn, torch)
    instance_state = _snapshot_module_state(trace_model, nn, torch)
    class_state = _snapshot_module_class_definitions(trace_model, nn)
    checked_call = _checked_call
    symbolic_trace = torch.fx.symbolic_trace
    trace_error: Exception | None = None
    traced: object | None = None
    try:
        if _security_context is None:
            traced = symbolic_trace(trace_model)
        else:
            if checked_call is None:
                raise UnsupportedOpError(
                    "Unsupported pipeline callback integrity adapter"
                )
            traced = checked_call(
                symbolic_trace,
                (trace_model,),
                {},
                nn,
                _security_context,
            )
    except Exception as exc:
        trace_error = exc
    class_state_changed = _restore_module_class_definitions(class_state)
    instance_state_changed = instance_state != _snapshot_module_state(
        trace_model,
        nn,
        torch,
    )
    if class_state_changed or instance_state_changed:
        raise UnsupportedOpError(
            "Unsupported Python state mutation during FX symbolic trace"
        )
    if trace_error is not None:
        raise UnsupportedOpError(
            f"Unsupported FX symbolic trace semantics: {trace_error}"
        ) from trace_error
    assert traced is not None
    traced, transformations = _fuse_conv_batchnorm_eval(traced, nn, torch)
    has_batchnorm_fusion = bool(transformations)
    modules = dict(_trusted_named_modules(traced, nn))
    graph = lower_fx_graph(
        model=model,
        traced=traced,
        modules=modules,
        input_shape=normalized_input_shape,
        float_dtype=float_dtype,
        transformations=transformations,
        nn=nn,
        torch=torch,
    )
    if has_batchnorm_fusion:
        _validate_fusion_boundary(
            source_model=model,
            fused_model=traced,
            input_shape=normalized_input_shape,
            float_dtype=float_dtype,
            nn=nn,
            torch=torch,
            security_context=_security_context,
            checked_call=checked_call,
        )
    _validate_lowered_semantics(
        model=traced if has_batchnorm_fusion else model,
        graph=graph,
        input_shape=normalized_input_shape,
        float_dtype=float_dtype,
        nn=nn,
        torch=torch,
        copy_model=not has_batchnorm_fusion,
        security_context=_security_context,
        checked_call=checked_call,
    )
    return graph
