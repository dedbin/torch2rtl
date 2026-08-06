from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


SEED = 42
TRAIN_SIZE = 90
VALIDATION_SIZE = 30
TEST_SIZE = 30


@dataclass(frozen=True)
class IrisData:
    train_features: pd.DataFrame
    validation_features: pd.DataFrame
    test_features: pd.DataFrame
    train_targets: torch.Tensor
    validation_targets: torch.Tensor
    test_targets: torch.Tensor
    train_inputs: torch.Tensor
    validation_inputs: torch.Tensor
    test_inputs: torch.Tensor
    train_min: torch.Tensor
    train_max: torch.Tensor
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]

    @property
    def train_indices(self) -> tuple[int, ...]:
        return tuple(int(index) for index in self.train_features.index)

    @property
    def validation_indices(self) -> tuple[int, ...]:
        return tuple(int(index) for index in self.validation_features.index)

    @property
    def test_indices(self) -> tuple[int, ...]:
        return tuple(int(index) for index in self.test_features.index)


def prepare_iris_data(seed: int = SEED) -> IrisData:
    frame, feature_names, target_names = _load_frame()
    features = frame.loc[:, list(feature_names)]
    targets = frame.loc[:, "target"]
    train_features, rest_features, train_targets, rest_targets = train_test_split(
        features,
        targets,
        test_size=0.4,
        random_state=seed,
        stratify=targets,
    )
    validation_features, test_features, validation_targets, test_targets = (
        train_test_split(
            rest_features,
            rest_targets,
            test_size=0.5,
            random_state=seed,
            stratify=rest_targets,
        )
    )
    return _build_data(
        train_features=train_features,
        validation_features=validation_features,
        test_features=test_features,
        train_target_series=train_targets,
        validation_target_series=validation_targets,
        test_target_series=test_targets,
        feature_names=feature_names,
        target_names=target_names,
    )


def restore_iris_data(checkpoint: Mapping[str, Any]) -> IrisData:
    frame, feature_names, target_names = _load_frame()
    _require_metadata(checkpoint, "feature_names", list(feature_names))
    _require_metadata(checkpoint, "target_names", list(target_names))

    train_indices = _checkpoint_indices(checkpoint, "train_indices", TRAIN_SIZE)
    validation_indices = _checkpoint_indices(
        checkpoint,
        "validation_indices",
        VALIDATION_SIZE,
    )
    test_indices = _checkpoint_indices(checkpoint, "test_indices", TEST_SIZE)
    _validate_index_partition(train_indices, validation_indices, test_indices, len(frame))

    features = frame.loc[:, list(feature_names)]
    targets = frame.loc[:, "target"]
    data = _build_data(
        train_features=features.loc[list(train_indices)].copy(),
        validation_features=features.loc[list(validation_indices)].copy(),
        test_features=features.loc[list(test_indices)].copy(),
        train_target_series=targets.loc[list(train_indices)].copy(),
        validation_target_series=targets.loc[list(validation_indices)].copy(),
        test_target_series=targets.loc[list(test_indices)].copy(),
        feature_names=feature_names,
        target_names=target_names,
    )

    checkpoint_min = _checkpoint_tensor(checkpoint, "train_min", len(feature_names))
    checkpoint_max = _checkpoint_tensor(checkpoint, "train_max", len(feature_names))
    if not torch.equal(data.train_min, checkpoint_min):
        raise ValueError("checkpoint train_min does not match the restored train split")
    if not torch.equal(data.train_max, checkpoint_max):
        raise ValueError("checkpoint train_max does not match the restored train split")
    return _replace_normalization(data, checkpoint_min, checkpoint_max)


def normalize_tensor(
    values: torch.Tensor,
    train_min: torch.Tensor,
    train_max: torch.Tensor,
) -> torch.Tensor:
    span = train_max - train_min
    if not bool(torch.all(span > 0)):
        raise ValueError("cannot min-max normalize a constant training feature")
    return 2.0 * (values - train_min) / span - 1.0


def _load_frame() -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    dataset = load_iris(as_frame=True)
    frame = dataset.frame
    if frame is None:
        raise RuntimeError("load_iris(as_frame=True) did not return a pandas frame")
    feature_names = tuple(str(name) for name in dataset.feature_names)
    target_names = tuple(str(name) for name in dataset.target_names)
    return frame.copy(), feature_names, target_names


def _build_data(
    *,
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_target_series: pd.Series,
    validation_target_series: pd.Series,
    test_target_series: pd.Series,
    feature_names: tuple[str, ...],
    target_names: tuple[str, ...],
) -> IrisData:
    _validate_split(train_features, train_target_series, TRAIN_SIZE, "train")
    _validate_split(
        validation_features,
        validation_target_series,
        VALIDATION_SIZE,
        "validation",
    )
    _validate_split(test_features, test_target_series, TEST_SIZE, "test")
    _validate_index_partition(
        tuple(int(index) for index in train_features.index),
        tuple(int(index) for index in validation_features.index),
        tuple(int(index) for index in test_features.index),
        TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE,
    )

    train_raw = _feature_tensor(train_features)
    validation_raw = _feature_tensor(validation_features)
    test_raw = _feature_tensor(test_features)
    train_min = train_raw.min(dim=0).values
    train_max = train_raw.max(dim=0).values
    return IrisData(
        train_features=train_features.copy(),
        validation_features=validation_features.copy(),
        test_features=test_features.copy(),
        train_targets=_target_tensor(train_target_series),
        validation_targets=_target_tensor(validation_target_series),
        test_targets=_target_tensor(test_target_series),
        train_inputs=normalize_tensor(train_raw, train_min, train_max),
        validation_inputs=normalize_tensor(validation_raw, train_min, train_max),
        test_inputs=normalize_tensor(test_raw, train_min, train_max),
        train_min=train_min,
        train_max=train_max,
        feature_names=feature_names,
        target_names=target_names,
    )


def _replace_normalization(
    data: IrisData,
    train_min: torch.Tensor,
    train_max: torch.Tensor,
) -> IrisData:
    return IrisData(
        train_features=data.train_features,
        validation_features=data.validation_features,
        test_features=data.test_features,
        train_targets=data.train_targets,
        validation_targets=data.validation_targets,
        test_targets=data.test_targets,
        train_inputs=normalize_tensor(
            _feature_tensor(data.train_features), train_min, train_max
        ),
        validation_inputs=normalize_tensor(
            _feature_tensor(data.validation_features), train_min, train_max
        ),
        test_inputs=normalize_tensor(
            _feature_tensor(data.test_features), train_min, train_max
        ),
        train_min=train_min.clone(),
        train_max=train_max.clone(),
        feature_names=data.feature_names,
        target_names=data.target_names,
    )


def _feature_tensor(features: pd.DataFrame) -> torch.Tensor:
    return torch.tensor(features.to_numpy(copy=True), dtype=torch.float32)


def _target_tensor(targets: pd.Series) -> torch.Tensor:
    return torch.tensor(targets.to_numpy(copy=True), dtype=torch.long)


def _validate_split(
    features: pd.DataFrame,
    targets: pd.Series,
    expected_size: int,
    name: str,
) -> None:
    if len(features) != expected_size or len(targets) != expected_size:
        raise ValueError(f"{name} split must contain exactly {expected_size} samples")
    counts = targets.value_counts().sort_index().tolist()
    expected_per_class = expected_size // 3
    if counts != [expected_per_class] * 3:
        raise ValueError(f"{name} split is not equally stratified across three classes")


def _validate_index_partition(
    train_indices: tuple[int, ...],
    validation_indices: tuple[int, ...],
    test_indices: tuple[int, ...],
    dataset_size: int,
) -> None:
    index_sets = tuple(map(set, (train_indices, validation_indices, test_indices)))
    if any(index_sets[left] & index_sets[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("train, validation, and test indices must not overlap")
    combined = index_sets[0] | index_sets[1] | index_sets[2]
    if combined != set(range(dataset_size)):
        raise ValueError("split indices must partition the complete Iris dataset")


def _checkpoint_indices(
    checkpoint: Mapping[str, Any],
    name: str,
    expected_size: int,
) -> tuple[int, ...]:
    values = checkpoint.get(name)
    if type(values) is not list or any(type(value) is not int for value in values):
        raise TypeError(f"checkpoint {name} must be a list of integers")
    if len(values) != expected_size:
        raise ValueError(f"checkpoint {name} must contain {expected_size} entries")
    return tuple(values)


def _checkpoint_tensor(
    checkpoint: Mapping[str, Any],
    name: str,
    feature_count: int,
) -> torch.Tensor:
    value = checkpoint.get(name)
    if (
        type(value) is not torch.Tensor
        or value.dtype != torch.float32
        or value.shape != (feature_count,)
    ):
        raise TypeError(
            f"checkpoint {name} must be a float32 tensor with shape "
            f"({feature_count},)"
        )
    return value.detach().cpu()


def _require_metadata(
    checkpoint: Mapping[str, Any],
    name: str,
    expected: list[str],
) -> None:
    value = checkpoint.get(name)
    if value != expected:
        raise ValueError(f"checkpoint {name} does not match sklearn Iris metadata")
