from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from torch2rtl.eda_tools import find_eda_tool
from torch2rtl.synth.report import update_report


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

    iverilog = find_eda_tool("iverilog")
    vvp = find_eda_tool("vvp")
    verilator = find_eda_tool("verilator")
    if iverilog and vvp:
        result = _run_icarus(build_dir, iverilog, vvp)
    elif verilator:
        result = _run_verilator(build_dir, verilator)
    else:
        result = SimulationResult(
            ok=False,
            status="not_found",
            message="simulator not found: install Icarus Verilog or Verilator",
        )
    update_report(build_dir, "simulation", result.__dict__)
    return result


def _run_icarus(build_dir: Path, iverilog: str, vvp: str) -> SimulationResult:
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
    ok = _simulation_ok(run_result.returncode, run_result.stdout)
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


def _run_verilator(build_dir: Path, verilator: str) -> SimulationResult:
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
    ok = _simulation_ok(run_result.returncode, run_result.stdout)
    return SimulationResult(
        ok=ok,
        status="passed" if ok else "failed",
        message="simulation passed" if ok else "simulation failed",
        stdout=run_result.stdout,
        stderr=run_result.stderr,
    )


def _simulation_ok(returncode: int, stdout: str) -> bool:
    lines = [line.strip() for line in stdout.splitlines()]
    has_pass = any(line.startswith("PASS") for line in lines)
    has_fail = any(line.startswith("FAIL") for line in lines)
    return returncode == 0 and has_pass and not has_fail


def _source_files(build_dir: Path) -> list[str]:
    optional = ["conv2d_comb.sv"]
    required = ["linear_comb.sv", "relu.sv", "argmax.sv", "top.sv", "tb_top.sv"]
    return [name for name in optional if (build_dir / name).exists()] + required
