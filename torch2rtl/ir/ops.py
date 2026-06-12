from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from torch2rtl.ir.tensor import TensorIR


@dataclass(frozen=True)
class LinearIR:
    name: str
    input: TensorIR
    output: TensorIR
    in_features: int
    out_features: int
    weight: np.ndarray
    bias: np.ndarray | None


@dataclass(frozen=True)
class ReluIR:
    name: str
    input: TensorIR
    output: TensorIR


@dataclass(frozen=True)
class FlattenIR:
    name: str
    input: TensorIR
    output: TensorIR


@dataclass(frozen=True)
class ArgmaxIR:
    name: str
    input: TensorIR
    output: TensorIR


OpIR: TypeAlias = LinearIR | ReluIR | FlattenIR | ArgmaxIR
