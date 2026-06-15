from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch2rtl.backend.systemverilog.emit import BuildReport, emit_systemverilog
from torch2rtl.eda_tools import detect_eda_environment, detect_eda_tools
from torch2rtl.frontend.pytorch_fx import load_model_from_file, parse_model
from torch2rtl.quant.fixed_point import FixedPointConfig
from torch2rtl.synth.report import update_report_metadata
from torch2rtl.synth.yosys import SynthResult, run_yosys
from torch2rtl.visualization import render_visualization_from_build
from torch2rtl.verify.simulator import SimulationResult, run_simulation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEMO_OUT = Path("build/demo")


@dataclass(frozen=True)
class DemoSpec:
    name: str
    model_path: Path
    input_shape: tuple[int, ...]
    description: str
    seed: int
    vectors: int = 16


@dataclass(frozen=True)
class DemoResult:
    name: str
    build_report: BuildReport
    simulation: SimulationResult
    synthesis: SynthResult
    visualization_path: Path

    @property
    def ok(self) -> bool:
        simulation_ok = self.simulation.ok or self.simulation.status == "not_found"
        synthesis_ok = self.synthesis.ok or self.synthesis.status == "not_found"
        return simulation_ok and synthesis_ok


DEMO_SPECS = {
    "tiny-mlp": DemoSpec(
        name="tiny-mlp",
        model_path=REPO_ROOT / "examples" / "tiny_mlp" / "model.py",
        input_shape=(16,),
        description="Small Linear/ReLU classifier over a 16-value vector.",
        seed=1,
    ),
    "grid-classifier": DemoSpec(
        name="grid-classifier",
        model_path=REPO_ROOT / "examples" / "grid_classifier" / "model.py",
        input_shape=(4, 4),
        description="Three-layer MLP over a 4x4 grid.",
        seed=7,
    ),
    "tiny-conv": DemoSpec(
        name="tiny-conv",
        model_path=REPO_ROOT / "examples" / "tiny_conv" / "model.py",
        input_shape=(1, 3, 3),
        description="Conv2d/ReLU/Linear classifier for the smallest CNN path.",
        seed=11,
    ),
    "image-cnn": DemoSpec(
        name="image-cnn",
        model_path=REPO_ROOT / "examples" / "image_cnn" / "model.py",
        input_shape=(1, 4, 4),
        description="Synthetic-image CNN; uses a checkpoint if one exists.",
        seed=2026,
    ),
}


def demo_names() -> tuple[str, ...]:
    return tuple(DEMO_SPECS)


def get_demo_spec(name: str) -> DemoSpec:
    try:
        return DEMO_SPECS[name]
    except KeyError as exc:
        available = ", ".join(demo_names())
        raise ValueError(f"unknown demo: {name}; choose one of: {available}") from exc


def run_demo(
    name: str,
    out_dir: Path = DEFAULT_DEMO_OUT,
    bits: int = 8,
    frac_bits: int = 6,
    acc_bits: int = 32,
    vectors: int | None = None,
    seed: int | None = None,
    vector_index: int = 0,
) -> DemoResult:
    spec = get_demo_spec(name)
    cfg = FixedPointConfig(bits=bits, frac_bits=frac_bits, acc_bits=acc_bits)
    vector_count = spec.vectors if vectors is None else vectors
    vector_seed = spec.seed if seed is None else seed

    model = load_model_from_file(spec.model_path)
    graph = parse_model(model, spec.input_shape)
    build_report = emit_systemverilog(
        graph=graph,
        cfg=cfg,
        out_dir=out_dir,
        vector_count=vector_count,
        seed=vector_seed,
    )
    update_report_metadata(
        out_dir,
        _demo_report_payload(
            spec=spec,
            out_dir=out_dir,
            cfg=cfg,
            vectors=vector_count,
            seed=vector_seed,
            vector_index=vector_index,
        ),
    )
    simulation = run_simulation(out_dir)
    synthesis = run_yosys(out_dir)
    visualization_path = render_visualization_from_build(
        build_dir=out_dir,
        vector_index=vector_index,
    )
    return DemoResult(
        name=spec.name,
        build_report=build_report,
        simulation=simulation,
        synthesis=synthesis,
        visualization_path=visualization_path,
    )


def _demo_report_payload(
    spec: DemoSpec,
    out_dir: Path,
    cfg: FixedPointConfig,
    vectors: int,
    seed: int,
    vector_index: int,
) -> dict[str, Any]:
    return {
        "demo": {
            "name": spec.name,
            "description": spec.description,
            "model_path": _relative_path(spec.model_path),
            "input_shape": list(spec.input_shape),
            "commands": _commands(spec, out_dir, cfg, vectors, seed, vector_index),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "tools": detect_eda_environment(),
        },
        "tools": detect_eda_tools(),
    }


def _commands(
    spec: DemoSpec,
    out_dir: Path,
    cfg: FixedPointConfig,
    vectors: int,
    seed: int,
    vector_index: int,
) -> list[dict[str, str]]:
    out = _format_path(out_dir)
    model = _relative_path(spec.model_path)
    shape = " ".join(str(dim) for dim in spec.input_shape)
    common_quant = (
        f"--bits {cfg.bits} --frac-bits {cfg.frac_bits} --acc-bits {cfg.acc_bits}"
    )
    return [
        {
            "step": "demo",
            "command": (
                f"torch2rtl demo --name {spec.name} --out {out} "
                f"{common_quant} --vectors {vectors} --seed {seed}"
            ),
        },
        {
            "step": "compile",
            "command": (
                f"torch2rtl compile {model} --input-shape {shape} "
                f"{common_quant} --vectors {vectors} --seed {seed} --out {out}"
            ),
        },
        {"step": "verify", "command": f"torch2rtl verify {out}"},
        {"step": "synth", "command": f"torch2rtl synth {out}"},
        {
            "step": "visualize",
            "command": f"torch2rtl visualize {out} --vector-index {vector_index}",
        },
    ]


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _format_path(path: Path) -> str:
    text = path.as_posix()
    if " " in text:
        return f'"{text}"'
    return text
