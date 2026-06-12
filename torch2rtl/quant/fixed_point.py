from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FixedPointConfig:
    bits: int = 8
    frac_bits: int = 6
    acc_bits: int = 32

    def __post_init__(self) -> None:
        if self.bits < 2:
            raise ValueError("bits must be at least 2 for signed fixed-point")
        if self.frac_bits < 0:
            raise ValueError("frac_bits must be non-negative")
        if self.acc_bits <= self.bits:
            raise ValueError("acc_bits must be larger than bits")

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
    scaled = np.rint(np.asarray(values, dtype=np.float64) * cfg.scale)
    return saturate_array(scaled, cfg.bits).astype(np.int64)


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
