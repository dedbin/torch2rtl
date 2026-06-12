from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from torch2rtl.ir.ops import LinearIR, OpIR
from torch2rtl.ir.tensor import TensorIR


@dataclass(frozen=True)
class GraphIR:
    input: TensorIR
    output: TensorIR
    ops: tuple[OpIR, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def linear_ops(self) -> tuple[LinearIR, ...]:
        return tuple(op for op in self.ops if isinstance(op, LinearIR))
