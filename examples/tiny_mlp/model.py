from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def create_model(checkpoint_path: Path | None = None) -> TinyMLP:
    model = TinyMLP()
    if checkpoint_path is not None:
        load_checkpoint(model, checkpoint_path)
    model.eval()
    return model


def load_checkpoint(model: TinyMLP, checkpoint_path: Path) -> None:
    checkpoint = _torch_load(checkpoint_path)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Invalid checkpoint format: {checkpoint_path}")
    model.load_state_dict(state_dict)


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, dict):
        raise TypeError(f"Invalid checkpoint payload: {path}")
    return loaded
