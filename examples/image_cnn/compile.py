from __future__ import annotations

import argparse
import logging
from pathlib import Path

from data import IMAGE_SIZE
from model import DEFAULT_CHECKPOINT_PATH, INPUT_CHANNELS, create_model

from torch2rtl.backend.systemverilog.emit import emit_systemverilog
from torch2rtl.frontend.pytorch_fx import parse_model
from torch2rtl.quant.fixed_point import FixedPointConfig


LOGGER = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--out", type=Path, default=Path("build/image_cnn/rtl"))
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--frac-bits", type=int, default=6)
    parser.add_argument("--acc-bits", type=int, default=32)
    parser.add_argument("--vectors", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. "
            "Run examples/image_cnn/train.py first."
        )

    model = create_model(checkpoint_path=args.checkpoint)
    graph = parse_model(model, input_shape=(INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
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
    LOGGER.info("compiled checkpoint %s -> %s", args.checkpoint, args.out)
    LOGGER.info("generated %d files", len(report.generated_files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
