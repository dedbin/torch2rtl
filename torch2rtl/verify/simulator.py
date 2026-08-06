from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from jinja2 import Environment, FileSystemLoader, select_autoescape

from torch2rtl.eda_tools import find_eda_tool
from torch2rtl.synth.report import (
    ReportValidationError,
    read_compile_report,
    record_preflight_failure,
    update_report,
)
from torch2rtl.verify.source_integrity import validate_rtl_source_hashes


_EXTERNAL_JSON_ERRORS = (
    OSError,
    UnicodeDecodeError,
    json.JSONDecodeError,
    ValueError,
    RecursionError,
)


@dataclass(frozen=True)
class SimulationResult:
    ok: bool
    status: str
    message: str
    stdout: str = ""
    stderr: str = ""


def run_simulation(build_dir: Path) -> SimulationResult:
    build_dir = build_dir.resolve()
    if not build_dir.exists():
        return SimulationResult(False, "error", f"build directory not found: {build_dir}")

    expected_vectors, vector_error = _validate_vector_files(build_dir)
    if vector_error is not None:
        result = SimulationResult(
            ok=False,
            status="invalid_vectors",
            message=vector_error,
        )
        try:
            update_report(build_dir, "simulation", result.__dict__)
        except (OSError, ReportValidationError):
            record_preflight_failure(build_dir, "simulation", result.__dict__)
        return result

    iverilog = find_eda_tool("iverilog")
    vvp = find_eda_tool("vvp")
    verilator = find_eda_tool("verilator")
    if iverilog and vvp:
        result = _run_icarus(build_dir, iverilog, vvp, expected_vectors)
    elif verilator:
        result = _run_verilator(build_dir, verilator, expected_vectors)
    else:
        result = SimulationResult(
            ok=False,
            status="not_found",
            message="simulator not found: install Icarus Verilog or Verilator",
        )
    update_report(build_dir, "simulation", result.__dict__)
    return result


def _run_icarus(
    build_dir: Path,
    iverilog: str,
    vvp: str,
    expected_vectors: int,
) -> SimulationResult:
    sources = _source_files(build_dir)
    output = "simv"
    compile_cmd = [iverilog, "-g2012", "-o", output, *sources]
    compile_result = subprocess.run(
        compile_cmd,
        cwd=build_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if compile_result.returncode != 0:
        return SimulationResult(
            ok=False,
            status="failed",
            message="iverilog compile failed",
            stdout=compile_result.stdout,
            stderr=compile_result.stderr,
        )
    _patch_vvp_shebang(build_dir / output, vvp)
    run_result = subprocess.run(
        [vvp, "-M", "-", output],
        cwd=build_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    ok = _simulation_ok(
        run_result.returncode,
        run_result.stdout,
        expected_vectors=expected_vectors,
    )
    return SimulationResult(
        ok=ok,
        status="passed" if ok else "failed",
        message="simulation passed" if ok else "simulation failed",
        stdout=run_result.stdout,
        stderr=run_result.stderr,
    )


def _patch_vvp_shebang(script_path: Path, vvp: str) -> None:
    if not script_path.exists():
        return
    lines = script_path.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("#!"):
        return
    runtime_path = _vvp_runtime_path(vvp)
    shebang = f"#! {runtime_path}"
    if lines[0] == shebang:
        return
    lines[0] = shebang
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _vvp_runtime_path(vvp: str) -> str:
    if _is_windows_path_text(vvp):
        return PureWindowsPath(vvp).as_posix()

    path = Path(vvp).resolve()
    if path.suffix.lower() in {".bat", ".cmd"}:
        suite_vvp = path.parent.parent / "oss-cad-suite" / "bin" / "vvp.exe"
        if suite_vvp.exists():
            path = suite_vvp
    return path.as_posix()


def _is_windows_path_text(path: str) -> bool:
    windows_path = PureWindowsPath(path)
    return bool(windows_path.drive) or "\\" in path


def _run_verilator(
    build_dir: Path,
    verilator: str,
    expected_vectors: int,
) -> SimulationResult:
    sources = _source_files(build_dir)
    cmd = [verilator, "--binary", "--timing", "-sv", *sources, "--top-module", "tb_top"]
    build_result = subprocess.run(
        cmd,
        cwd=build_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if build_result.returncode != 0:
        return SimulationResult(
            ok=False,
            status="failed",
            message="verilator build failed",
            stdout=build_result.stdout,
            stderr=build_result.stderr,
        )
    executable = build_dir / "obj_dir" / "Vtb_top"
    if not executable.exists():
        executable = build_dir / "obj_dir" / "Vtb_top.exe"
    run_result = subprocess.run(
        [str(executable)],
        cwd=build_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    ok = _simulation_ok(
        run_result.returncode,
        run_result.stdout,
        expected_vectors=expected_vectors,
    )
    return SimulationResult(
        ok=ok,
        status="passed" if ok else "failed",
        message="simulation passed" if ok else "simulation failed",
        stdout=run_result.stdout,
        stderr=run_result.stderr,
    )


def _simulation_ok(
    returncode: int,
    stdout: str,
    expected_vectors: int | None = None,
) -> bool:
    lines = [line.strip() for line in stdout.splitlines()]
    pass_counts = [
        int(match.group(1))
        for line in lines
        if (match := re.fullmatch(r"PASS vectors=(\d+)", line)) is not None
    ]
    has_fail = any(line.startswith("FAIL") for line in lines)
    if returncode != 0 or has_fail or len(pass_counts) != 1 or pass_counts[0] <= 0:
        return False
    return expected_vectors is None or pass_counts[0] == expected_vectors


def _validate_vector_files(build_dir: Path) -> tuple[int, str | None]:
    metadata_path = build_dir / "vectors.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except _EXTERNAL_JSON_ERRORS as exc:
        return 0, f"invalid vector metadata: {exc}"
    if not isinstance(metadata, dict):
        return 0, "invalid vector metadata: root must be a JSON object"

    expected: dict[str, int] = {}
    for name in ("count", "input_size", "logits_size", "class_count", "data_bits"):
        value = metadata.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return 0, f"invalid vector metadata: {name} must be a positive integer"
        expected[name] = value
    if not 2 <= expected["data_bits"] <= 32:
        return 0, "invalid vector metadata: data_bits must be in range 2..32"
    if expected["class_count"] != expected["logits_size"]:
        return 0, "invalid vector metadata: class_count must equal logits_size"

    testbench, testbench_error = _read_testbench_contract(build_dir)
    if testbench_error is not None:
        return 0, testbench_error
    assert testbench is not None
    compile_error = _validate_compile_contract(build_dir, expected, testbench)
    if compile_error is not None:
        return 0, compile_error
    metadata_to_testbench = {
        "count": "EXPECTED_VECTORS",
        "input_size": "INPUT_SIZE",
        "logits_size": "CLASS_COUNT",
        "class_count": "CLASS_COUNT",
        "data_bits": "DATA_BITS",
    }
    for metadata_name, testbench_name in metadata_to_testbench.items():
        if expected[metadata_name] != testbench[testbench_name]:
            return 0, (
                f"invalid vector metadata: {metadata_name}={expected[metadata_name]} "
                f"does not match testbench {testbench_name}={testbench[testbench_name]}"
            )

    file_counts = {
        "input_vectors.txt": expected["count"] * expected["input_size"],
        "expected_classes.txt": expected["count"],
        "expected_logits.txt": expected["count"] * expected["logits_size"],
    }
    data_min = -(1 << (expected["data_bits"] - 1))
    data_max = (1 << (expected["data_bits"] - 1)) - 1
    file_ranges = {
        "input_vectors.txt": (data_min, data_max),
        "expected_classes.txt": (0, expected["class_count"] - 1),
        "expected_logits.txt": (data_min, data_max),
    }
    for filename, required_count in file_counts.items():
        path = build_dir / filename
        try:
            tokens = path.read_text(encoding="utf-8").split()
        except (UnicodeDecodeError, OSError) as exc:
            return 0, f"invalid vector file {filename}: {exc}"
        if len(tokens) != required_count:
            return 0, (
                f"invalid vector file {filename}: expected {required_count} values, "
                f"got {len(tokens)}"
            )
        try:
            values = [int(token, 10) for token in tokens]
        except ValueError:
            return 0, f"invalid vector file {filename}: non-integer value"
        if any(token != str(value) for token, value in zip(tokens, values, strict=True)):
            return 0, (
                f"invalid vector file {filename}: values must use canonical signed "
                "decimal notation"
            )
        minimum, maximum = file_ranges[filename]
        if any(value < minimum or value > maximum for value in values):
            return 0, (
                f"invalid vector file {filename}: values must be in "
                f"range [{minimum}, {maximum}]"
            )
    return expected["count"], None


def _read_testbench_contract(
    build_dir: Path,
) -> tuple[dict[str, int] | None, str | None]:
    testbench_path = build_dir / "tb_top.sv"
    try:
        text = testbench_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        return None, f"invalid testbench contract: {exc}"
    code, lexical_error = _mask_systemverilog_non_code(text)
    if lexical_error is not None:
        return None, f"invalid testbench contract: {lexical_error}"
    if "`" in code:
        return None, (
            "invalid testbench contract: SystemVerilog preprocessor directives "
            "are not supported"
        )

    header = re.match(
        r"\A\s*module\s+tb_top\s*;\s*"
        r"localparam\s+int\s+DATA_BITS\s*=\s*(\d+)\s*;\s*"
        r"localparam\s+int\s+FRAC_BITS\s*=\s*(\d+)\s*;\s*"
        r"localparam\s+int\s+ACC_BITS\s*=\s*(\d+)\s*;\s*"
        r"localparam\s+int\s+INPUT_SIZE\s*=\s*(\d+)\s*;\s*"
        r"localparam\s+int\s+CLASS_COUNT\s*=\s*(\d+)\s*;\s*"
        r"localparam\s+int\s+CLASS_BITS\s*=\s*(\d+)\s*;\s*"
        r"localparam\s+int\s+EXPECTED_VECTORS\s*=\s*(\d+)\s*;\s*"
        r"localparam\s+longint\s+signed\s+DATA_MIN\s*=\s*-64'sd(\d+)\s*;\s*"
        r"localparam\s+longint\s+signed\s+DATA_MAX\s*=\s*64'sd(\d+)\s*;",
        code,
    )
    if header is None:
        return None, (
            "invalid testbench contract: expected the generated top-level "
            "parameter header immediately after module tb_top"
        )
    try:
        (
            data_bits,
            frac_bits,
            acc_bits,
            input_size,
            class_count,
            class_bits,
            expected_vectors,
            data_min_abs,
            data_max,
        ) = (int(value) for value in header.groups())
    except ValueError:
        return None, (
            "invalid testbench contract: numeric parameter is not a supported integer"
        )
    values = {
        "DATA_BITS": data_bits,
        "FRAC_BITS": frac_bits,
        "ACC_BITS": acc_bits,
        "INPUT_SIZE": input_size,
        "CLASS_COUNT": class_count,
        "CLASS_BITS": class_bits,
        "EXPECTED_VECTORS": expected_vectors,
    }
    for name in ("DATA_BITS", "INPUT_SIZE", "CLASS_COUNT", "EXPECTED_VECTORS"):
        value = values[name]
        if value <= 0:
            return None, f"invalid testbench contract: {name} must be positive"
    if not 2 <= values["DATA_BITS"] <= 32:
        return None, "invalid testbench contract: DATA_BITS must be in range 2..32"
    if not 0 <= frac_bits < data_bits:
        return None, "invalid testbench contract: FRAC_BITS must satisfy 0 <= F < DATA_BITS"
    if not data_bits < acc_bits <= 64:
        return None, (
            "invalid testbench contract: ACC_BITS must satisfy "
            "DATA_BITS < ACC_BITS <= 64"
        )
    expected_class_bits = max(1, (class_count - 1).bit_length())
    if class_bits != expected_class_bits:
        return None, (
            f"invalid testbench contract: CLASS_BITS={class_bits} does not match "
            f"CLASS_COUNT={class_count}"
        )
    if data_min_abs != 1 << (data_bits - 1) or data_max != (1 << (data_bits - 1)) - 1:
        return None, (
            "invalid testbench contract: DATA_MIN/DATA_MAX do not match DATA_BITS"
        )
    expected_source = _render_generated_testbench(
        data_bits=data_bits,
        frac_bits=frac_bits,
        acc_bits=acc_bits,
        input_size=input_size,
        class_count=class_count,
        class_bits=class_bits,
        expected_vectors=expected_vectors,
        data_min_abs=data_min_abs,
        data_max=data_max,
    )
    if text != expected_source:
        return None, (
            "invalid testbench contract: checker body does not match the "
            "generated template"
        )
    return values, None


def _validate_compile_contract(
    build_dir: Path,
    vectors: dict[str, int],
    testbench: dict[str, int],
) -> str | None:
    dut_error = _validate_dut_sources(build_dir)
    if dut_error is not None:
        return dut_error

    try:
        report = read_compile_report(build_dir / "report.json")
    except ReportValidationError as exc:
        return str(exc)

    quant = report.get("quant")
    assert isinstance(quant, dict)
    expected_quant = {
        "bits": testbench["DATA_BITS"],
        "frac_bits": testbench["FRAC_BITS"],
        "acc_bits": testbench["ACC_BITS"],
    }
    for name, expected_value in expected_quant.items():
        value = quant.get(name)
        if type(value) is not int or value != expected_value:
            return (
                f"invalid compile report: quant.{name}={value!r} does not match "
                f"testbench value {expected_value}"
            )

    report_vectors = report.get("vectors")
    assert isinstance(report_vectors, dict)
    if report_vectors["count"] != vectors["count"]:
        return "invalid compile report: vector count does not match vectors.json"

    graph = report.get("graph")
    input_shape = graph.get("input_shape") if isinstance(graph, dict) else None
    if not isinstance(input_shape, list) or any(
        type(dimension) is not int or dimension <= 0 for dimension in input_shape
    ):
        return "invalid compile report: graph.input_shape must contain positive integers"
    report_input_size = 1
    for dimension in input_shape:
        report_input_size *= dimension
    if report_input_size != vectors["input_size"]:
        return "invalid compile report: input size does not match vectors.json"

    metrics = report.get("metrics")
    activation_values = metrics.get("activation_values") if isinstance(metrics, dict) else None
    if (
        not isinstance(activation_values, list)
        or not activation_values
        or type(activation_values[-1]) is not int
        or activation_values[-1] != vectors["logits_size"]
    ):
        return "invalid compile report: logits size does not match vectors.json"

    top_path = build_dir / "top.sv"
    try:
        top_text = top_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        return f"invalid compiled top contract: {exc}"
    top_code, lexical_error = _mask_systemverilog_non_code(top_text)
    if lexical_error is not None or "`" in top_code:
        return "invalid compiled top contract: unsupported lexical structure"
    top_header = re.match(
        r"\A\s*module\s+top\s*#\s*\(\s*"
        r"parameter\s+int\s+DATA_BITS\s*=\s*(\d+)\s*,\s*"
        r"parameter\s+int\s+FRAC_BITS\s*=\s*(\d+)\s*,\s*"
        r"parameter\s+int\s+ACC_BITS\s*=\s*(\d+)\s*,\s*"
        r"parameter\s+int\s+INPUT_SIZE\s*=\s*(\d+)\s*,\s*"
        r"parameter\s+int\s+CLASS_COUNT\s*=\s*(\d+)\s*,\s*"
        r"parameter\s+int\s+CLASS_BITS\s*=\s*(\d+)\s*\)",
        top_code,
    )
    if top_header is None:
        return "invalid compiled top contract: generated parameter header is missing"
    try:
        top_values = tuple(int(value) for value in top_header.groups())
    except ValueError:
        return "invalid compiled top contract: numeric parameter is not a supported integer"
    expected_top = (
        testbench["DATA_BITS"],
        testbench["FRAC_BITS"],
        testbench["ACC_BITS"],
        testbench["INPUT_SIZE"],
        testbench["CLASS_COUNT"],
        testbench["CLASS_BITS"],
    )
    if top_values != expected_top:
        return "invalid compiled top contract: parameters do not match testbench"
    integrity_error = validate_rtl_source_hashes(build_dir, report)
    if integrity_error is not None:
        return integrity_error
    return None


def _validate_dut_sources(build_dir: Path) -> str | None:
    for filename in _source_files(build_dir):
        if filename == "tb_top.sv":
            continue
        path = build_dir / filename
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            return f"invalid DUT source {filename}: {exc}"
        code, lexical_error = _mask_systemverilog_non_code(text)
        if lexical_error is not None:
            return f"invalid DUT source {filename}: {lexical_error}"
        if "`" in code:
            return f"invalid DUT source {filename}: preprocessor directives are forbidden"
        if "$" in code or re.search(
            r"\b(?:initial|final|force|release|fork|join|join_any|join_none)\b",
            code,
        ):
            return (
                f"invalid DUT source {filename}: simulation-control constructs "
                "are forbidden outside the generated testbench"
            )
    return None


def _render_generated_testbench(**context: int) -> str:
    template_dir = Path(__file__).parents[1] / "backend" / "systemverilog" / "templates"
    environment = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(disabled_extensions=("sv", "j2")),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return environment.get_template("tb_top.sv.j2").render(**context)


def _mask_systemverilog_non_code(text: str) -> tuple[str, str | None]:
    """Mask comments and strings while preserving code positions and newlines."""
    masked = list(text)
    state = "code"
    index = 0
    while index < len(text):
        current = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if state == "code":
            if current == "/" and following == "/":
                masked[index] = masked[index + 1] = " "
                state = "line_comment"
                index += 2
                continue
            if current == "/" and following == "*":
                masked[index] = masked[index + 1] = " "
                state = "block_comment"
                index += 2
                continue
            if current == '"':
                masked[index] = " "
                state = "string"
                index += 1
                continue
        elif state == "line_comment":
            if current in "\r\n":
                state = "code"
            else:
                masked[index] = " "
        elif state == "block_comment":
            if current == "*" and following == "/":
                masked[index] = masked[index + 1] = " "
                state = "code"
                index += 2
                continue
            if current not in "\r\n":
                masked[index] = " "
        else:
            if current == "\\":
                masked[index] = " "
                if following:
                    if following not in "\r\n":
                        masked[index + 1] = " "
                    index += 2
                    continue
            elif current == '"':
                masked[index] = " "
                state = "code"
            elif current in "\r\n":
                return "", "unterminated string literal"
            else:
                masked[index] = " "
        index += 1

    if state == "block_comment":
        return "", "unterminated block comment"
    if state == "string":
        return "", "unterminated string literal"
    return "".join(masked), None


def _source_files(build_dir: Path) -> list[str]:
    optional = ["conv2d_comb.sv"]
    required = ["linear_comb.sv", "relu.sv", "argmax.sv", "top.sv", "tb_top.sv"]
    return [name for name in optional if (build_dir / name).exists()] + required
