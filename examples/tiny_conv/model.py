from __future__ import annotations

import torch
import torch.nn as nn


INPUT_CHANNELS = 1
GRID_ROWS = 3
GRID_COLS = 3
CONV_CHANNELS = 1
CONV_ROWS = 2
CONV_COLS = 2
CLASS_COUNT = 4


class TinyConvClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                INPUT_CHANNELS,
                CONV_CHANNELS,
                kernel_size=2,
            ),
            nn.ReLU(),
            nn.Flatten(start_dim=0),
            nn.Linear(CONV_CHANNELS * CONV_ROWS * CONV_COLS, CLASS_COUNT),
        )
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _initialize_weights(self) -> None:
        with torch.no_grad():
            conv = self.net[0]
            linear = self.net[3]
            if not isinstance(conv, nn.Conv2d) or not isinstance(linear, nn.Linear):
                raise TypeError("TinyConvClassifier weight layout changed")

            conv.weight.zero_()
            conv.bias.zero_()
            conv.weight[0, 0] = torch.tensor(
                [
                    [0.50, 0.25],
                    [0.25, 0.50],
                ],
                dtype=conv.weight.dtype,
            )

            linear.weight.zero_()
            linear.bias.zero_()
            for class_idx in range(CLASS_COUNT):
                linear.weight[class_idx, class_idx] = 1.0


def create_model() -> TinyConvClassifier:
    return TinyConvClassifier()
