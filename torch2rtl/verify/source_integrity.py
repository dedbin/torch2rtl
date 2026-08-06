from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from torch2rtl.synth.report import ReportValidationError, read_compile_report


RTL_SOURCE_NAMES = (
    "argmax.sv",
    "conv2d_comb.sv",
    "linear_comb.sv",
    "relu.sv",
    "tb_top.sv",
    "top.sv",
)

_EXTERNAL_JSON_ERRORS = (
    OSError,
    UnicodeDecodeError,
    json.JSONDecodeError,
    ValueError,
    RecursionError,
)


def validate_rtl_source_hashes(
    build_dir: Path,
    report: dict[str, object],
) -> str | None:
    recorded = report.get("rtl_sha256")
    expected_names = set(RTL_SOURCE_NAMES)
    if not isinstance(recorded, dict) or set(recorded) != expected_names:
        return "invalid RTL source integrity: compile report hash manifest is incomplete"
    for filename in RTL_SOURCE_NAMES:
        digest = recorded.get(filename)
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            return f"invalid RTL source integrity: malformed hash for {filename}"
        try:
            actual = hashlib.sha256((build_dir / filename).read_bytes()).hexdigest()
        except OSError as exc:
            return f"invalid RTL source integrity: cannot read {filename}: {exc}"
        if actual != digest:
            return f"invalid RTL source integrity: {filename} differs from compile output"
    return None


def validate_build_rtl_sources(build_dir: Path) -> str | None:
    try:
        report = read_compile_report(build_dir / "report.json")
    except ReportValidationError as exc:
        return f"invalid synthesis compile contract: {exc}"
    integrity_error = validate_rtl_source_hashes(build_dir, report)
    if integrity_error is not None:
        return integrity_error
    return _validate_synthesis_compile_contract(build_dir, report)


def _validate_synthesis_compile_contract(
    build_dir: Path,
    report: dict[str, object],
) -> str | None:
    try:
        top_text = (build_dir / "top.sv").read_text(encoding="utf-8")
        testbench_text = (build_dir / "tb_top.sv").read_text(encoding="utf-8")
        vectors = json.loads((build_dir / "vectors.json").read_text(encoding="utf-8"))
    except _EXTERNAL_JSON_ERRORS as exc:
        return f"invalid synthesis compile contract: cannot read metadata: {exc}"
    if not isinstance(vectors, dict):
        return "invalid synthesis compile contract: vectors metadata must be an object"

    top = _extract_declarations(top_text, "parameter", (
        "DATA_BITS",
        "FRAC_BITS",
        "ACC_BITS",
        "INPUT_SIZE",
        "CLASS_COUNT",
        "CLASS_BITS",
    ))
    testbench = _extract_declarations(testbench_text, "localparam", (
        "DATA_BITS",
        "FRAC_BITS",
        "ACC_BITS",
        "INPUT_SIZE",
        "CLASS_COUNT",
        "CLASS_BITS",
        "EXPECTED_VECTORS",
    ))
    if top is None or testbench is None:
        return (
            "invalid synthesis compile contract: generated width header is missing "
            "or has an unsupported numeric value"
        )
    for name in ("DATA_BITS", "FRAC_BITS", "ACC_BITS", "INPUT_SIZE", "CLASS_COUNT", "CLASS_BITS"):
        if top[name] != testbench[name]:
            return "invalid synthesis compile contract: top and testbench widths differ"

    quant = report.get("quant")
    if not isinstance(quant, dict):
        return "invalid synthesis compile contract: report quant metadata is missing"
    for report_name, header_name in (
        ("bits", "DATA_BITS"),
        ("frac_bits", "FRAC_BITS"),
        ("acc_bits", "ACC_BITS"),
    ):
        value = quant.get(report_name)
        if type(value) is not int or value != top[header_name]:
            return "invalid synthesis compile contract: report quant widths differ from RTL"

    vector_fields = {
        "count": testbench["EXPECTED_VECTORS"],
        "input_size": top["INPUT_SIZE"],
        "logits_size": top["CLASS_COUNT"],
        "class_count": top["CLASS_COUNT"],
        "data_bits": top["DATA_BITS"],
    }
    for name, expected in vector_fields.items():
        value = vectors.get(name)
        if type(value) is not int or value != expected:
            return "invalid synthesis compile contract: vectors metadata differs from RTL"

    report_vectors = report.get("vectors")
    if (
        not isinstance(report_vectors, dict)
        or type(report_vectors.get("count")) is not int
        or report_vectors["count"] != vectors["count"]
    ):
        return "invalid synthesis compile contract: report vector count differs"
    graph = report.get("graph")
    input_shape = graph.get("input_shape") if isinstance(graph, dict) else None
    if not isinstance(input_shape, list) or any(
        type(dimension) is not int or dimension <= 0 for dimension in input_shape
    ):
        return "invalid synthesis compile contract: report input shape is invalid"
    input_size = 1
    for dimension in input_shape:
        input_size *= dimension
    if input_size != top["INPUT_SIZE"]:
        return "invalid synthesis compile contract: report input size differs from RTL"
    metrics = report.get("metrics")
    activations = metrics.get("activation_values") if isinstance(metrics, dict) else None
    if (
        not isinstance(activations, list)
        or not activations
        or type(activations[-1]) is not int
        or activations[-1] != top["CLASS_COUNT"]
    ):
        return "invalid synthesis compile contract: report logits size differs from RTL"
    return None


def _extract_declarations(
    text: str,
    keyword: str,
    names: tuple[str, ...],
) -> dict[str, int] | None:
    values: dict[str, int] = {}
    for name in names:
        match = re.search(
            rf"\b{keyword}\s+int\s+{name}\s*=\s*(\d+)\s*[,;)]",
            text,
        )
        if match is None:
            return None
        try:
            values[name] = int(match.group(1))
        except ValueError:
            return None
    return values
