from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


IMAGE_SIZE = 4
CLASS_NAMES = ("horizontal", "vertical", "diag_down", "diag_up")
CLASS_COUNT = len(CLASS_NAMES)


@dataclass(frozen=True)
class DatasetConfig:
    samples_per_class: int = 64
    noise: float = 0.08
    seed: int = 2026


def generate_dataset(cfg: DatasetConfig) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(cfg.seed)
    images: list[np.ndarray] = []
    labels: list[int] = []

    for class_id in range(CLASS_COUNT):
        for _ in range(cfg.samples_per_class):
            images.append(_pattern_image(class_id, rng, cfg.noise))
            labels.append(class_id)

    order = rng.permutation(len(labels))
    image_array = np.asarray(images, dtype=np.float32)[order]
    label_array = np.asarray(labels, dtype=np.int64)[order]
    return (
        torch.from_numpy(image_array[:, None, :, :]),
        torch.from_numpy(label_array),
    )


def _pattern_image(class_id: int, rng: np.random.Generator, noise: float) -> np.ndarray:
    image = rng.normal(loc=0.0, scale=noise, size=(IMAGE_SIZE, IMAGE_SIZE)).astype(
        np.float32
    )
    image += 0.08 * rng.random((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    if class_id == 0:
        row = int(rng.integers(1, IMAGE_SIZE - 1))
        image[row, :] += 1.0
        image[row - 1, :] += 0.35
        image[row + 1, :] += 0.35
    elif class_id == 1:
        col = int(rng.integers(1, IMAGE_SIZE - 1))
        image[:, col] += 1.0
        image[:, col - 1] += 0.35
        image[:, col + 1] += 0.35
    elif class_id == 2:
        offset = int(rng.integers(-1, 2))
        _draw_diagonal(image, offset=offset, descending=True)
    elif class_id == 3:
        offset = int(rng.integers(-1, 2))
        _draw_diagonal(image, offset=offset, descending=False)
    else:
        raise ValueError(f"Unknown class_id: {class_id}")

    return np.clip(image, 0.0, 1.0)


def _draw_diagonal(image: np.ndarray, offset: int, descending: bool) -> None:
    for row in range(IMAGE_SIZE):
        col = row + offset if descending else (IMAGE_SIZE - 1 - row) + offset
        if 0 <= col < IMAGE_SIZE:
            image[row, col] += 1.0
        if 0 <= col - 1 < IMAGE_SIZE:
            image[row, col - 1] += 0.25
        if 0 <= col + 1 < IMAGE_SIZE:
            image[row, col + 1] += 0.25
