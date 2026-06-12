from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TensorIR:
    name: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total
