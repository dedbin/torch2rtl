from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from data import CLASS_COUNT, IMAGE_SIZE


INPUT_CHANNELS = 1
CONV_CHANNELS = 4
DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[2] / "build" / "image_cnn" / "image_cnn.pt"
)


class ImageCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=INPUT_CHANNELS,
                out_channels=CONV_CHANNELS,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Flatten(start_dim=0),
            nn.Linear(CONV_CHANNELS * IMAGE_SIZE * IMAGE_SIZE, CLASS_COUNT),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def create_model(checkpoint_path: Path | None = None) -> ImageCNN:
    model = ImageCNN()
    path = DEFAULT_CHECKPOINT_PATH if checkpoint_path is None else checkpoint_path
    if path.exists():
        load_checkpoint(model, path)
    model.eval()
    return model


def load_checkpoint(model: ImageCNN, checkpoint_path: Path) -> None:
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
