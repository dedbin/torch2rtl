from __future__ import annotations

import argparse
from pathlib import Path

from model import create_model

from torch2rtl.backend.systemverilog.emit import emit_systemverilog
from torch2rtl.frontend.pytorch_fx import parse_model
from torch2rtl.quant.fixed_point import FixedPointConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("build"))
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--frac-bits", type=int, default=6)
    parser.add_argument("--acc-bits", type=int, default=32)
    args = parser.parse_args()

    model = create_model()
    graph = parse_model(model, input_shape=(16,))
    cfg = FixedPointConfig(
        bits=args.bits,
        frac_bits=args.frac_bits,
        acc_bits=args.acc_bits,
    )
    report = emit_systemverilog(graph, cfg, args.out)
    print(f"generated {len(report.generated_files)} files in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
