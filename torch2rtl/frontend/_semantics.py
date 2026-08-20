from __future__ import annotations

import math

import numpy as np
import torch2rtl.quant.reference as _quant_reference

from torch2rtl.frontend._errors import UnsupportedOpError
from torch2rtl.frontend._state_guard import (
    _copy_model_for_tracing,
    _restore_module_class_definitions,
    _snapshot_module_class_definitions,
    _snapshot_module_state,
)
from torch2rtl.ir.graph import GraphIR


_INFER_FLOAT_GRAPH = _quant_reference.infer_float_graph


def _run_concrete_probe(
    model: object,
    probe_input: object,
    label: str,
    nn: object,
    torch: object,
    *,
    copy_model: bool = True,
    security_context: tuple[object, ...] | None = None,
    checked_call: object | None = None,
) -> object:
    probe_model = _copy_model_for_tracing(model, nn, torch) if copy_model else model
    before_state = _snapshot_module_state(probe_model, nn, torch)
    class_state = _snapshot_module_class_definitions(probe_model, nn)
    probe_error: Exception | None = None
    output: object | None = None
    try:
        if security_context is None:
            is_grad_enabled = torch.is_grad_enabled
            set_grad_enabled = torch._C._set_grad_enabled
            previous_grad_mode = is_grad_enabled()
            set_grad_enabled(False)
            try:
                output = probe_model(probe_input)
            finally:
                set_grad_enabled(previous_grad_mode)
        else:
            if checked_call is None:
                raise UnsupportedOpError(
                    "Unsupported semantic probe integrity adapter"
                )
            output = checked_call(
                probe_model,
                (probe_input,),
                {},
                nn,
                security_context,
                disable_grad=True,
            )
    except Exception as exc:
        probe_error = exc
    class_changed = _restore_module_class_definitions(class_state)
    instance_changed = before_state != _snapshot_module_state(
        probe_model,
        nn,
        torch,
    )
    if class_changed or instance_changed:
        raise UnsupportedOpError(
            f"Unsupported Python state mutation during {label} semantic probe"
        )
    if probe_error is not None:
        raise UnsupportedOpError(
            f"Unsupported {label} model semantics: {probe_error}"
        ) from probe_error
    return output


def _validate_fusion_boundary(
    source_model: object,
    fused_model: object,
    input_shape: tuple[int, ...],
    float_dtype: str,
    nn: object,
    torch: object,
    security_context: tuple[object, ...] | None = None,
    *,
    checked_call: object | None = None,
) -> None:
    dtype = torch.float32 if float_dtype == "float32" else torch.float64
    numpy_dtype = np.dtype(float_dtype)
    element_count = math.prod(input_shape)
    probe_values = (
        np.zeros(input_shape, dtype=numpy_dtype),
        np.linspace(-0.75, 0.75, num=element_count, dtype=numpy_dtype).reshape(
            input_shape
        ),
    )
    rtol, atol = (
        (1e-5, 1e-6) if float_dtype == "float32" else (1e-12, 1e-12)
    )
    for values in probe_values:
        probe_input = torch.from_numpy(values.copy()).to(dtype=dtype)
        source_output = _run_concrete_probe(
            source_model,
            probe_input.unsqueeze(0),
            "source singleton-batch",
            nn,
            torch,
            security_context=security_context,
            checked_call=checked_call,
        )
        fused_output = _run_concrete_probe(
            fused_model,
            probe_input,
            "fused batchless",
            nn,
            torch,
            copy_model=False,
            security_context=security_context,
            checked_call=checked_call,
        )
        if not isinstance(source_output, torch.Tensor) or not isinstance(
            fused_output,
            torch.Tensor,
        ):
            raise UnsupportedOpError(
                "Unsupported fusion semantics: source and fused models must "
                "return one Tensor"
            )
        if source_output.dtype != fused_output.dtype:
            raise UnsupportedOpError(
                "Unsupported fusion semantics: source and fused output dtypes "
                f"differ ({source_output.dtype} != {fused_output.dtype})"
            )

        source_values = source_output.detach().cpu().numpy()
        fused_values = fused_output.detach().cpu().numpy()
        if source_values.shape == () or fused_values.shape == ():
            if (
                source_values.shape != ()
                or fused_values.shape != ()
                or source_output.dtype is not torch.int64
                or source_values.item() != fused_values.item()
            ):
                raise UnsupportedOpError(
                    "Unsupported fusion semantics: scalar Argmax/class_id "
                    "changed across Conv2d/BatchNorm2d fusion"
                )
            continue

        if source_output.dtype is not dtype or fused_output.dtype is not dtype:
            raise UnsupportedOpError(
                "Unsupported fusion semantics: tensor outputs must preserve "
                f"the model dtype {dtype}"
            )
        if source_values.shape[0] != 1:
            raise UnsupportedOpError(
                "Unsupported fusion semantics: source tensor output must have "
                "a leading singleton batch dimension"
            )
        source_batchless = source_values[0]
        if source_batchless.shape != fused_values.shape:
            raise UnsupportedOpError(
                "Unsupported fusion semantics: source singleton-batch output "
                f"shape {source_batchless.shape} does not match fused batchless "
                f"shape {fused_values.shape}"
            )
        if not np.all(np.isfinite(source_batchless)) or not np.all(
            np.isfinite(fused_values)
        ):
            raise UnsupportedOpError(
                "Unsupported fusion semantics: source and fused outputs must "
                "contain only finite values"
            )
        if not np.allclose(
            source_batchless,
            fused_values,
            rtol=rtol,
            atol=atol,
        ):
            raise UnsupportedOpError(
                "Unsupported fusion semantics: source singleton-batch output "
                "does not match fused batchless output"
            )
        if int(np.argmax(source_batchless.reshape(-1))) != int(
            np.argmax(fused_values.reshape(-1))
        ):
            raise UnsupportedOpError(
                "Unsupported fusion semantics: Argmax/class_id changed across "
                "Conv2d/BatchNorm2d fusion"
            )


def _validate_lowered_semantics(
    model: object,
    graph: GraphIR,
    input_shape: tuple[int, ...],
    float_dtype: str,
    nn: object,
    torch: object,
    *,
    copy_model: bool = True,
    security_context: tuple[object, ...] | None = None,
    checked_call: object | None = None,
) -> None:
    dtype = torch.float32 if float_dtype == "float32" else torch.float64
    numpy_dtype = np.dtype(float_dtype)
    element_count = math.prod(input_shape)
    probe_values = (
        np.zeros(input_shape, dtype=numpy_dtype),
        np.linspace(-0.75, 0.75, num=element_count, dtype=numpy_dtype).reshape(
            input_shape
        ),
    )
    for values in probe_values:
        probe_input = torch.from_numpy(values.copy()).to(dtype=dtype)
        actual = _run_concrete_probe(
            model,
            probe_input,
            "concrete",
            nn,
            torch,
            copy_model=copy_model,
            security_context=security_context,
            checked_call=checked_call,
        )
        if not isinstance(actual, torch.Tensor):
            raise UnsupportedOpError(
                "Unsupported model output: concrete forward must return one Tensor"
            )

        expected_dtype = torch.int64 if graph.output.dtype == "int64" else dtype
        if actual.dtype != expected_dtype:
            raise UnsupportedOpError(
                "Unsupported model output dtype: concrete forward produced "
                f"{actual.dtype}, GraphIR requires {expected_dtype}"
            )
        actual_values = actual.detach().cpu().numpy()
        expected_values = _INFER_FLOAT_GRAPH(graph, values).output
        if actual_values.shape != expected_values.shape or not np.array_equal(
            actual_values,
            expected_values,
            equal_nan=True,
        ):
            raise UnsupportedOpError(
                "Unsupported model semantics: concrete forward does not match lowered GraphIR"
            )
