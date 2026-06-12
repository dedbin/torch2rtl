from __future__ import annotations

import os
from pathlib import Path
from shutil import which

EDA_TOOL_NAMES = ("iverilog", "vvp", "verilator", "yosys")


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


def _local_tool_candidates(name: str) -> tuple[Path, ...]:
    tool_dir = Path(__file__).resolve().parents[1] / "tools" / "eda-bin"
    suffixes = _executable_suffixes()
    return tuple(tool_dir / f"{name}{suffix}" for suffix in suffixes)


def _executable_suffixes() -> tuple[str, ...]:
    if os.name == "nt":
        return ("", ".cmd", ".bat", ".exe")
    return ("",)
