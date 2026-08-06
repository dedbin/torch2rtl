from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class FixedPointConfig:
    bits: int = 8
    frac_bits: int = 6
    acc_bits: int = 32

    def __post_init__(self) -> None:
        for name in ("bits", "frac_bits", "acc_bits"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
            object.__setattr__(self, name, int(value))
        if not 2 <= self.bits <= 32:
            raise ValueError("bits must be in the supported signed range 2..32")
        if not 0 <= self.frac_bits < self.bits:
            raise ValueError("frac_bits must satisfy 0 <= frac_bits < bits")
        if not self.bits < self.acc_bits <= 64:
            raise ValueError("acc_bits must satisfy bits < acc_bits <= 64")

    @property
    def scale(self) -> int:
        return 1 << self.frac_bits

    @property
    def min_int(self) -> int:
        return -(1 << (self.bits - 1))

    @property
    def max_int(self) -> int:
        return (1 << (self.bits - 1)) - 1

    @property
    def acc_min_int(self) -> int:
        return -(1 << (self.acc_bits - 1))

    @property
    def acc_max_int(self) -> int:
        return (1 << (self.acc_bits - 1)) - 1


def quantize_array(values: np.ndarray, cfg: FixedPointConfig) -> np.ndarray:
    source = np.asarray(values)
    if source.dtype.kind in "iu":
        source = source.astype(np.float64)
    elif source.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(
            "Quantization supports only float32 and float64 values, "
            f"got {source.dtype}"
        )
    if not np.all(np.isfinite(source)):
        raise ValueError("Cannot quantize NaN or infinity")
    scale = np.asarray(cfg.scale, dtype=source.dtype)
    # A finite source may overflow its floating dtype during scaling.  That
    # overflow is expected to saturate below, so it must not leak as a warning.
    with np.errstate(over="ignore"):
        scaled = np.rint(source * scale)
    # float32 cannot represent 2**31 - 1 exactly.  Clip in float64 after
    # dtype-faithful rounding so the signed 32-bit endpoint stays exact.
    return saturate_array(scaled.astype(np.float64), cfg.bits).astype(np.int64)


def dequantize_array(values: np.ndarray, cfg: FixedPointConfig) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) / float(cfg.scale)


def saturate_array(values: np.ndarray, bits: int) -> np.ndarray:
    min_int = -(1 << (bits - 1))
    max_int = (1 << (bits - 1)) - 1
    return np.clip(values, min_int, max_int)


def saturate_int(value: int, bits: int) -> int:
    min_int = -(1 << (bits - 1))
    max_int = (1 << (bits - 1)) - 1
    return max(min(value, max_int), min_int)
