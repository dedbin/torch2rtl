from __future__ import annotations

import argparse
from pathlib import Path

from model import create_model

from torch2rtl.backend.systemverilog.emit import emit_systemverilog
from torch2rtl.frontend.pytorch_fx import parse_model
from torch2rtl.quant.fixed_point import FixedPointConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("build"))
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--frac-bits", type=int, default=6)
    parser.add_argument("--acc-bits", type=int, default=32)
    parser.add_argument("--vectors", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. "
            "Run examples/tiny_mlp/train.py first."
        )
    model = create_model(checkpoint_path=args.checkpoint)
    graph = parse_model(model, input_shape=(16,))
    cfg = FixedPointConfig(
        bits=args.bits,
        frac_bits=args.frac_bits,
        acc_bits=args.acc_bits,
    )
    report = emit_systemverilog(
        graph,
        cfg,
        args.out,
        vector_count=args.vectors,
        seed=args.seed,
    )
    print(f"compiled checkpoint {args.checkpoint} -> {args.out}")
    print(f"generated {len(report.generated_files)} files in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
