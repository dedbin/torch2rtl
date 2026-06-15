from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from string import ascii_uppercase
from typing import Any

from torch2rtl.eda_tools import find_eda_tool
from torch2rtl.synth.report import update_report


@dataclass(frozen=True)
class SynthResult:
    ok: bool
    status: str
    message: str
    stdout: str = ""
    stderr: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


RESOURCE_LABELS = {
    "wires": "wires",
    "wire bits": "wire_bits",
    "public wires": "public_wires",
    "public wire bits": "public_wire_bits",
    "ports": "ports",
    "port bits": "port_bits",
    "memories": "memories",
    "memory bits": "memory_bits",
    "processes": "processes",
    "cells": "cells",
}


def run_yosys(build_dir: Path) -> SynthResult:
    build_dir = build_dir.resolve()
    if not build_dir.exists():
        return SynthResult(False, "error", f"build directory not found: {build_dir}")
    yosys = find_eda_tool("yosys")
    if yosys is None:
        result = SynthResult(
            ok=False,
            status="not_found",
            message="yosys not found: install Yosys to run synthesis",
        )
        update_report(build_dir, "synthesis", result.__dict__)
        return result

    sources = " ".join(_source_files(build_dir))
    script = f"read_verilog -sv {sources}; prep -top top; stat"
    run_result = _run_yosys(build_dir, yosys, script)
    full_log = run_result.stdout + run_result.stderr
    log_path = build_dir / "yosys.log"
    log_path.write_text(full_log, encoding="utf-8")
    result = SynthResult(
        ok=run_result.returncode == 0,
        status="passed" if run_result.returncode == 0 else "failed",
        message="synthesis passed" if run_result.returncode == 0 else "synthesis failed",
        stdout=_report_stdout(run_result.stdout),
        stderr=_tail_text(run_result.stderr),
        metrics=parse_yosys_metrics(full_log),
    )
    update_report(build_dir, "synthesis", result.__dict__)
    return result


def parse_yosys_metrics(text: str) -> dict[str, Any]:
    section = _last_design_hierarchy_section(text)
    metrics: dict[str, Any] = {}
    cell_types: dict[str, int] = {}

    for line in section.splitlines():
        stripped = line.strip()
        match = re.fullmatch(r"(-|\d+)\s+(.+)", stripped)
        if match is None:
            continue
        count = _yosys_count(match.group(1))
        label = match.group(2).strip()
        metric_name = RESOURCE_LABELS.get(label)
        if metric_name is not None:
            metrics[metric_name] = count
            continue
        if label.startswith("$"):
            cell_types[label] = count

    if cell_types:
        metrics["cell_types"] = cell_types
    return metrics


def _last_design_hierarchy_section(text: str) -> str:
    marker = "=== design hierarchy ==="
    index = text.rfind(marker)
    if index == -1:
        return text
    return text[index + len(marker) :]


def _yosys_count(value: str) -> int:
    if value == "-":
        return 0
    return int(value)


def _run_yosys(
    build_dir: Path,
    yosys: str,
    script: str,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [yosys, "-p", script],
        cwd=build_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if os.name != "nt" or "GetShortPathName() failed" not in result.stderr:
        return result
    return _run_yosys_with_subst(build_dir, script, result)


def _run_yosys_with_subst(
    build_dir: Path,
    script: str,
    fallback: subprocess.CompletedProcess[str],
) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[2]
    suite_root = repo_root / "tools" / "oss-cad-suite"
    yosys_exe = suite_root / "bin" / "yosys.exe"
    if not yosys_exe.exists():
        return fallback

    try:
        relative_build = build_dir.relative_to(repo_root)
    except ValueError:
        return fallback

    drive = _free_subst_drive()
    if drive is None:
        return fallback

    map_result = subprocess.run(
        ["subst", drive, str(repo_root)],
        capture_output=True,
        text=True,
        check=False,
    )
    if map_result.returncode != 0:
        return fallback

    try:
        mapped_root = Path(f"{drive}\\")
        mapped_suite = mapped_root / "tools" / "oss-cad-suite"
        env = _oss_cad_env(mapped_suite)
        return subprocess.run(
            [str(mapped_suite / "bin" / "yosys.exe"), "-p", script],
            cwd=mapped_root / relative_build,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        subprocess.run(
            ["subst", drive, "/D"],
            capture_output=True,
            text=True,
            check=False,
        )


def _free_subst_drive() -> str | None:
    for letter in reversed(ascii_uppercase):
        drive = f"{letter}:"
        if not Path(f"{drive}\\").exists():
            return drive
    return None


def _oss_cad_env(suite_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    suite = str(suite_root) + "\\"
    env["YOSYSHQ_ROOT"] = suite
    env["SSL_CERT_FILE"] = str(suite_root / "etc" / "cacert.pem")
    env["PATH"] = f"{suite_root / 'bin'};{suite_root / 'lib'};{env.get('PATH', '')}"
    env["PYTHON_EXECUTABLE"] = str(suite_root / "lib" / "python3.exe")
    env["YOSYS_DATDIR"] = str(suite_root / "share" / "yosys")
    env["QT_PLUGIN_PATH"] = str(suite_root / "lib" / "qt5" / "plugins")
    env["QT_LOGGING_RULES"] = "*=false"
    env["GTK_EXE_PREFIX"] = suite
    env["GTK_DATA_PREFIX"] = suite
    env["GDK_PIXBUF_MODULEDIR"] = str(
        suite_root / "lib" / "gdk-pixbuf-2.0" / "2.10.0" / "loaders"
    )
    env["GDK_PIXBUF_MODULE_FILE"] = str(
        suite_root / "lib" / "gdk-pixbuf-2.0" / "2.10.0" / "loaders.cache"
    )
    env["OPENFPGALOADER_SOJ_DIR"] = str(suite_root / "share" / "openFPGALoader")
    return env


def _report_stdout(stdout: str) -> str:
    marker = "=== design hierarchy ==="
    index = stdout.rfind(marker)
    if index == -1:
        return _tail_text(stdout)
    return _tail_text(stdout[index:])


def _tail_text(text: str, max_chars: int = 8000) -> str:
    if len(text) <= max_chars:
        return text
    return "... truncated ...\n" + text[-max_chars:]


def _source_files(build_dir: Path) -> list[str]:
    optional = ["conv2d_comb.sv"]
    required = ["linear_comb.sv", "relu.sv", "argmax.sv", "top.sv"]
    return [name for name in optional if (build_dir / name).exists()] + required
