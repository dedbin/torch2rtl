from __future__ import annotations

import argparse
from pathlib import Path

from torch2rtl.backend.systemverilog.emit import emit_systemverilog
from torch2rtl.demo import DEFAULT_DEMO_OUT, demo_names, run_demo
from torch2rtl.frontend.pytorch_fx import load_model_from_file, parse_model
from torch2rtl.quant.fixed_point import FixedPointConfig
from torch2rtl.synth.yosys import run_yosys
from torch2rtl.visualization import render_visualization_from_build
from torch2rtl.verify.simulator import run_simulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="torch2rtl",
        description="Compile a small PyTorch FX subset to fixed-point SystemVerilog.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser("compile", help="Compile a model file.")
    compile_parser.add_argument("model_path", type=Path)
    compile_parser.add_argument("--input-shape", type=int, nargs="+", required=True)
    compile_parser.add_argument("--bits", type=int, default=8)
    compile_parser.add_argument("--frac-bits", type=int, default=6)
    compile_parser.add_argument("--acc-bits", type=int, default=32)
    compile_parser.add_argument("--out", type=Path, default=Path("build"))
    compile_parser.add_argument("--vectors", type=int, default=16)
    compile_parser.add_argument("--seed", type=int, default=0)
    compile_parser.set_defaults(func=cmd_compile)

    verify_parser = subparsers.add_parser("verify", help="Run generated RTL tests.")
    verify_parser.add_argument("build_dir", type=Path)
    verify_parser.set_defaults(func=cmd_verify)

    synth_parser = subparsers.add_parser("synth", help="Run optional Yosys synthesis.")
    synth_parser.add_argument("build_dir", type=Path)
    synth_parser.set_defaults(func=cmd_synth)

    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Render the self-contained circuit visualization HTML.",
    )
    visualize_parser.add_argument("build_dir", type=Path)
    visualize_parser.add_argument("--out", type=Path, default=None)
    visualize_parser.add_argument("--vector-index", type=int, default=0)
    visualize_parser.set_defaults(func=cmd_visualize)

    demo_parser = subparsers.add_parser(
        "demo",
        help="Run a board-free FPGA demo flow.",
    )
    demo_parser.add_argument("demo_name", nargs="?", choices=demo_names())
    demo_parser.add_argument("--name", choices=demo_names(), default=None)
    demo_parser.add_argument("--out", type=Path, default=DEFAULT_DEMO_OUT)
    demo_parser.add_argument("--bits", type=int, default=8)
    demo_parser.add_argument("--frac-bits", type=int, default=6)
    demo_parser.add_argument("--acc-bits", type=int, default=32)
    demo_parser.add_argument("--vectors", type=int, default=None)
    demo_parser.add_argument("--seed", type=int, default=None)
    demo_parser.add_argument("--vector-index", type=int, default=0)
    demo_parser.set_defaults(func=cmd_demo)

    return parser


def cmd_compile(args: argparse.Namespace) -> int:
    cfg = FixedPointConfig(
        bits=args.bits,
        frac_bits=args.frac_bits,
        acc_bits=args.acc_bits,
    )
    model = load_model_from_file(args.model_path)
    graph = parse_model(model, tuple(args.input_shape))
    report = emit_systemverilog(
        graph=graph,
        cfg=cfg,
        out_dir=args.out,
        vector_count=args.vectors,
        seed=args.seed,
    )
    print(f"compiled {args.model_path} -> {args.out}")
    print(f"generated {len(report.generated_files)} files")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    result = run_simulation(args.build_dir)
    print(result.message)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return 0 if result.ok or result.status == "not_found" else 1


def cmd_synth(args: argparse.Namespace) -> int:
    result = run_yosys(args.build_dir)
    print(result.message)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return 0 if result.ok or result.status == "not_found" else 1


def cmd_visualize(args: argparse.Namespace) -> int:
    try:
        output = render_visualization_from_build(
            build_dir=args.build_dir,
            out_path=args.out,
            vector_index=args.vector_index,
        )
    except (FileNotFoundError, IndexError, ValueError) as exc:
        print(exc)
        return 1
    print(f"wrote visualization -> {output}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    name = args.name or args.demo_name
    if name is None:
        print(f"choose demo with --name; available: {', '.join(demo_names())}")
        return 1
    if args.name is not None and args.demo_name is not None and args.name != args.demo_name:
        print("--name and positional demo name must match")
        return 1
    result = run_demo(
        name=name,
        out_dir=args.out,
        bits=args.bits,
        frac_bits=args.frac_bits,
        acc_bits=args.acc_bits,
        vectors=args.vectors,
        seed=args.seed,
        vector_index=args.vector_index,
    )
    print(f"demo {result.name} -> {result.build_report.out_dir}")
    print(f"simulation: {result.simulation.message}")
    print(f"synthesis: {result.synthesis.message}")
    print(f"html report: {result.visualization_path}")
    return 0 if result.ok else 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
