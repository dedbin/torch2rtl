from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


INPUT_SIZE = 4
HIDDEN_SIZE = 8
OUTPUT_SIZE = 3
PARAMETER_COUNT = (
    INPUT_SIZE * HIDDEN_SIZE
    + HIDDEN_SIZE
    + HIDDEN_SIZE * OUTPUT_SIZE
    + OUTPUT_SIZE
)
DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[2] / "build" / "iris_mlp" / "iris_mlp.pt"
)


class IrisMLP(nn.Module):
    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        output_size: int = OUTPUT_SIZE,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: IrisMLP
    payload: dict[str, Any]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def load_model_checkpoint(path: Path) -> LoadedCheckpoint:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if type(payload) is not dict:
        raise TypeError("Iris checkpoint root must be a dictionary")
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise TypeError("Iris checkpoint model_state_dict must be a dictionary")

    model = IrisMLP(
        input_size=_checkpoint_size(payload, "input_size", INPUT_SIZE),
        hidden_size=_checkpoint_size(payload, "hidden_size", HIDDEN_SIZE),
        output_size=_checkpoint_size(payload, "output_size", OUTPUT_SIZE),
    )
    incompatible = model.load_state_dict(state_dict)
    missing_keys = tuple(incompatible.missing_keys)
    unexpected_keys = tuple(incompatible.unexpected_keys)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "checkpoint state_dict mismatch: "
            f"missing={missing_keys}, unexpected={unexpected_keys}"
        )
    for name, parameter in model.state_dict().items():
        expected = state_dict[name]
        if type(expected) is not torch.Tensor or not torch.equal(parameter, expected):
            raise RuntimeError(f"checkpoint parameter {name} did not load exactly")
    model.eval()
    return LoadedCheckpoint(model, payload, missing_keys, unexpected_keys)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _checkpoint_size(payload: dict[str, Any], name: str, expected: int) -> int:
    value = payload.get(name)
    if type(value) is not int or value != expected:
        raise ValueError(f"checkpoint {name} must equal {expected}")
    return value
