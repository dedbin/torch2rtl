from __future__ import annotations

import argparse
from pathlib import Path

from model import GRID_COLS, GRID_ROWS, create_model

from torch2rtl.backend.systemverilog.emit import emit_systemverilog
from torch2rtl.frontend.pytorch_fx import parse_model
from torch2rtl.quant.fixed_point import FixedPointConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("build/grid_classifier"))
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--frac-bits", type=int, default=6)
    parser.add_argument("--acc-bits", type=int, default=32)
    parser.add_argument("--vectors", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    model = create_model()
    graph = parse_model(model, input_shape=(GRID_ROWS, GRID_COLS))
    cfg = FixedPointConfig(
        bits=args.bits,
        frac_bits=args.frac_bits,
        acc_bits=args.acc_bits,
    )
    report = emit_systemverilog(
        graph=graph,
        cfg=cfg,
        out_dir=args.out,
        vector_count=args.vectors,
        seed=args.seed,
    )
    print(f"generated {len(report.generated_files)} files in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
