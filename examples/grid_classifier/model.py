from __future__ import annotations

import torch
import torch.nn as nn


GRID_ROWS = 4
GRID_COLS = 4
INPUT_SIZE = GRID_ROWS * GRID_COLS
EDGE_FEATURES = 8
HIDDEN_FEATURES = 8
CLASS_COUNT = 4


class GridClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(start_dim=0),
            nn.Linear(INPUT_SIZE, EDGE_FEATURES),
            nn.ReLU(),
            nn.Linear(EDGE_FEATURES, HIDDEN_FEATURES),
            nn.ReLU(),
            nn.Linear(HIDDEN_FEATURES, CLASS_COUNT),
        )
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _initialize_weights(self) -> None:
        with torch.no_grad():
            first = self.net[1]
            second = self.net[3]
            third = self.net[5]
            if not all(
                isinstance(layer, nn.Linear) for layer in (first, second, third)
            ):
                raise TypeError("GridClassifier weight layout changed")

            first.weight.zero_()
            first.bias.zero_()
            self._set_sum_feature(first, 0, [(0, col) for col in range(GRID_COLS)])
            self._set_sum_feature(
                first,
                1,
                [(GRID_ROWS - 1, col) for col in range(GRID_COLS)],
            )
            self._set_sum_feature(first, 2, [(row, 0) for row in range(GRID_ROWS)])
            self._set_sum_feature(
                first,
                3,
                [(row, GRID_COLS - 1) for row in range(GRID_ROWS)],
            )
            self._set_sum_feature(first, 4, [(idx, idx) for idx in range(GRID_ROWS)])
            self._set_sum_feature(
                first,
                5,
                [(idx, GRID_COLS - 1 - idx) for idx in range(GRID_ROWS)],
            )
            self._set_sum_feature(first, 6, [(1, 1), (1, 2), (2, 1), (2, 2)])
            self._set_sum_feature(
                first,
                7,
                [
                    (row, col)
                    for row in range(GRID_ROWS)
                    for col in range(GRID_COLS)
                ],
                scale=0.25,
            )

            second.weight.copy_(
                torch.tensor(
                    [
                        [1.00, -0.25, 0.00, 0.00, 0.00, 0.00, 0.50, 0.00],
                        [-0.25, 1.00, 0.00, 0.00, 0.00, 0.00, 0.50, 0.00],
                        [0.00, 0.00, 1.00, -0.25, 0.00, 0.00, 0.50, 0.00],
                        [0.00, 0.00, -0.25, 1.00, 0.00, 0.00, 0.50, 0.00],
                        [0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.25],
                        [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.25],
                        [0.50, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                        [0.00, 0.00, 0.50, 0.50, 0.00, 0.00, 0.00, 0.00],
                    ],
                    dtype=second.weight.dtype,
                )
            )
            second.bias.zero_()

            third.weight.copy_(
                torch.tensor(
                    [
                        [1.00, -0.30, 0.00, 0.00, 0.15, 0.00, 0.20, 0.00],
                        [-0.30, 1.00, 0.00, 0.00, 0.00, 0.15, 0.20, 0.00],
                        [0.00, 0.00, 1.00, -0.30, 0.15, 0.00, 0.00, 0.20],
                        [0.00, 0.00, -0.30, 1.00, 0.00, 0.15, 0.00, 0.20],
                    ],
                    dtype=third.weight.dtype,
                )
            )
            third.bias.zero_()

    @staticmethod
    def _set_sum_feature(
        layer: nn.Linear,
        feature_idx: int,
        cells: list[tuple[int, int]],
        scale: float = 1.0,
    ) -> None:
        for row, col in cells:
            layer.weight[feature_idx, row * GRID_COLS + col] = scale


def create_model() -> GridClassifier:
    return GridClassifier()
