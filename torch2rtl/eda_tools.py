from __future__ import annotations

import os
import subprocess
from pathlib import Path
from shutil import which

EDA_TOOL_NAMES = ("iverilog", "vvp", "verilator", "yosys")
VERSION_ARGS = {
    "iverilog": ("-V",),
    "vvp": ("-V",),
    "verilator": ("--version",),
    "yosys": ("-V",),
}


def find_eda_tool(name: str) -> str | None:
    found = which(name)
    if found is not None:
        return found

    for candidate in _local_tool_candidates(name):
        if candidate.exists():
            return str(candidate)
    return None


def detect_eda_tools() -> dict[str, bool]:
    return {name: find_eda_tool(name) is not None for name in EDA_TOOL_NAMES}


def detect_eda_environment() -> dict[str, dict[str, str | bool]]:
    return {name: _tool_environment(name) for name in EDA_TOOL_NAMES}


def _tool_environment(name: str) -> dict[str, str | bool]:
    path = find_eda_tool(name)
    if path is None:
        return {"available": False, "path": "", "version": ""}
    return {
        "available": True,
        "path": path,
        "version": _tool_version(name, path),
    }


def _tool_version(name: str, path: str) -> str:
    args = VERSION_ARGS.get(name, ("--version",))
    try:
        result = subprocess.run(
            [path, *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    lines = [line.strip() for line in (result.stdout + result.stderr).splitlines()]
    return next((line for line in lines if line), "")


def _local_tool_candidates(name: str) -> tuple[Path, ...]:
    tool_dir = Path(__file__).resolve().parents[1] / "tools" / "eda-bin"
    suffixes = _executable_suffixes()
    return tuple(tool_dir / f"{name}{suffix}" for suffix in suffixes)


def _executable_suffixes() -> tuple[str, ...]:
    if os.name == "nt":
        return ("", ".cmd", ".bat", ".exe")
    return ("",)
