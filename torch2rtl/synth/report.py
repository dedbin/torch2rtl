from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch2rtl.eda_tools import detect_eda_tools


class ReportValidationError(ValueError):
    """Raised when report.json is not a valid v0.2 compile report."""


_EXTERNAL_JSON_ERRORS = (
    OSError,
    UnicodeDecodeError,
    json.JSONDecodeError,
    ValueError,
    RecursionError,
)


def update_report(build_dir: Path, section: str, payload: dict[str, Any]) -> None:
    report_path = build_dir / "report.json"
    report = read_compile_report(report_path)
    report["tools"] = detect_eda_tools()
    status = report["status"]
    status[section] = payload
    report[section] = payload
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        if payload.get("status") == "invalid_vectors":
            _refresh_visualization_status(build_dir)
        else:
            _refresh_visualization(build_dir)
    except Exception:
        # report.json is authoritative for EDA; visualization is best-effort.
        pass


def update_report_metadata(build_dir: Path, payload: dict[str, Any]) -> None:
    report_path = build_dir / "report.json"
    report = read_compile_report(report_path)
    report.update(payload)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _refresh_visualization(build_dir)


def record_preflight_failure(
    build_dir: Path,
    section: str,
    payload: dict[str, Any],
) -> None:
    """Persist a failure without treating a malformed report as trusted metadata."""
    report_path = build_dir / "report.json"
    try:
        loaded = json.loads(report_path.read_text(encoding="utf-8"))
    except _EXTERNAL_JSON_ERRORS:
        loaded = {}
    report = loaded if type(loaded) is dict else {}
    generated_files = report.get("generated_files")
    if type(generated_files) is not list or any(
        type(name) is not str for name in generated_files
    ):
        # Status-only visualization refresh must never iterate untrusted report
        # metadata. Without this field it falls back to the existing manifest.
        report.pop("generated_files", None)
    report["status"] = {section: payload}
    report[section] = payload
    report["preflight_failure"] = {
        "section": section,
        "message": str(payload.get("message", "invalid compile report")),
    }
    try:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except (OSError, ValueError, RecursionError):
        pass
    try:
        _refresh_visualization_status(build_dir)
    except Exception:
        # The API/CLI failure remains authoritative even if a damaged optional
        # visualization cannot be refreshed.
        pass


def read_compile_report(report_path: Path) -> dict[str, Any]:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except _EXTERNAL_JSON_ERRORS as exc:
        raise ReportValidationError(
            f"invalid compile report: cannot read report.json: {exc}"
        ) from exc
    error = _validate_compile_report_metadata(report)
    if error is not None:
        raise ReportValidationError(f"invalid compile report: {error}")
    return report


def _validate_compile_report_metadata(report: object) -> str | None:
    if type(report) is not dict:
        return "root must be a JSON object"

    preflight_failure = report.get("preflight_failure")
    if preflight_failure is not None:
        return "contains a recorded preflight failure; regenerate the build directory"

    status = report.get("status")
    if type(status) is not dict:
        return "status must be a JSON object"
    for section, payload in status.items():
        if type(section) is not str or type(payload) is not dict:
            return "status entries must map strings to JSON objects"
        for name, expected_type in (
            ("ok", bool),
            ("status", str),
            ("message", str),
            ("stdout", str),
            ("stderr", str),
        ):
            if type(payload.get(name)) is not expected_type:
                return f"status.{section}.{name} has an invalid type"

    quant = report.get("quant")
    if type(quant) is not dict:
        return "quant must be a JSON object"
    quant_values: dict[str, int] = {}
    for name in ("bits", "frac_bits", "acc_bits"):
        value = quant.get(name)
        if type(value) is not int:
            return f"quant.{name} must be an integer"
        quant_values[name] = value
    bits = quant_values["bits"]
    frac_bits = quant_values["frac_bits"]
    acc_bits = quant_values["acc_bits"]
    if not 2 <= bits <= 32:
        return "quant.bits must be in range 2..32"
    if not 0 <= frac_bits < bits:
        return "quant.frac_bits must satisfy 0 <= frac_bits < bits"
    if not bits < acc_bits <= 64:
        return "quant.acc_bits must satisfy bits < acc_bits <= 64"

    vectors = report.get("vectors")
    if type(vectors) is not dict:
        return "vectors must be a JSON object"
    vector_count = vectors.get("count")
    if type(vector_count) is not int or vector_count <= 0:
        return "vectors.count must be a positive integer"

    if type(report.get("tool")) is not str or report["tool"] != "torch2rtl":
        return "tool must be the string 'torch2rtl'"
    if type(report.get("version")) is not str or report["version"] != "0.2.0":
        return "version must be the string '0.2.0'"

    graph = report.get("graph")
    if type(graph) is not dict:
        return "graph must be a JSON object"
    input_shape = graph.get("input_shape")
    if not _is_integer_shape(input_shape, allow_scalar=False):
        return "graph.input_shape must be a non-empty list of positive integers"
    output_shape = graph.get("output_shape")
    if not _is_integer_shape(output_shape, allow_scalar=True):
        return "graph.output_shape must be a list of positive integers"
    ops = graph.get("ops")
    if type(ops) is not list or any(type(name) is not str for name in ops):
        return "graph.ops must be a list of strings"

    metrics = report.get("metrics")
    if type(metrics) is not dict:
        return "metrics must be a JSON object"
    for name in ("parameters", "macs"):
        value = metrics.get(name)
        if type(value) is not int or value < 0:
            return f"metrics.{name} must be a non-negative integer"
    activation_values = metrics.get("activation_values")
    if (
        type(activation_values) is not list
        or not activation_values
        or any(type(value) is not int or value <= 0 for value in activation_values)
    ):
        return "metrics.activation_values must contain positive integers"
    metric_ops = metrics.get("ops")
    if type(metric_ops) is not list or any(type(item) is not dict for item in metric_ops):
        return "metrics.ops must be a list of JSON objects"

    if type(report.get("reference")) is not dict:
        return "reference must be a JSON object"
    generated_files = report.get("generated_files")
    if (
        type(generated_files) is not list
        or not generated_files
        or any(type(name) is not str for name in generated_files)
    ):
        return "generated_files must be a non-empty list of strings"
    if type(report.get("rtl_sha256")) is not dict:
        return "rtl_sha256 must be a JSON object"
    tools = report.get("tools")
    required_tools = {"iverilog", "vvp", "verilator", "yosys"}
    if (
        type(tools) is not dict
        or not required_tools <= set(tools)
        or any(type(value) is not bool for value in tools.values())
    ):
        return "tools must be a JSON object with boolean values"
    return None


def _is_integer_shape(value: object, *, allow_scalar: bool) -> bool:
    return (
        type(value) is list
        and (allow_scalar or bool(value))
        and all(type(dimension) is int and dimension > 0 for dimension in value)
    )


def _refresh_visualization(build_dir: Path) -> None:
    manifest_path = build_dir / "visualization.json"
    if not manifest_path.exists():
        return

    from torch2rtl.visualization.manifest import render_visualization_from_build

    try:
        render_visualization_from_build(build_dir)
    except (IndexError, OSError, UnicodeDecodeError, ValueError):
        _refresh_visualization_status(build_dir)


def _refresh_visualization_status(build_dir: Path) -> None:
    from torch2rtl.visualization.manifest import refresh_visualization_report_status

    refresh_visualization_report_status(build_dir)
