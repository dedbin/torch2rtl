from __future__ import annotations

import argparse
import copy
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

if __package__:
    from .data import IrisData, SEED, prepare_iris_data, restore_iris_data
    from .model import (
        DEFAULT_CHECKPOINT_PATH,
        HIDDEN_SIZE,
        INPUT_SIZE,
        OUTPUT_SIZE,
        IrisMLP,
        load_model_checkpoint,
    )
else:
    from data import IrisData, SEED, prepare_iris_data, restore_iris_data
    from model import (
        DEFAULT_CHECKPOINT_PATH,
        HIDDEN_SIZE,
        INPUT_SIZE,
        OUTPUT_SIZE,
        IrisMLP,
        load_model_checkpoint,
    )


LEARNING_RATE = 0.01
MAX_EPOCHS = 300


@dataclass(frozen=True)
class TrainingResult:
    model: IrisMLP
    data: IrisData
    best_epoch: int
    best_validation_loss: float
    train_accuracy: float
    validation_accuracy: float
    test_accuracy: float
    confusion: np.ndarray
    checkpoint_path: Path


def train_iris(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    *,
    epochs: int = MAX_EPOCHS,
    seed: int = SEED,
) -> TrainingResult:
    if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= MAX_EPOCHS:
        raise ValueError(f"epochs must be an integer in range 1..{MAX_EPOCHS}")
    _set_seed(seed)
    data = prepare_iris_data(seed)
    model = IrisMLP()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_epoch = 0
    best_validation_loss = float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_logits = model(data.train_inputs)
        train_loss = loss_fn(train_logits, data.train_targets)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = loss_fn(
                model(data.validation_inputs),
                data.validation_targets,
            )
        if validation_loss.item() < best_validation_loss:
            best_epoch = epoch
            best_validation_loss = float(validation_loss.item())
            best_state_dict = copy.deepcopy(model.state_dict())

    if best_state_dict is None:
        raise RuntimeError("training did not produce a best validation checkpoint")
    model.load_state_dict(best_state_dict)
    model.eval()
    with torch.no_grad():
        train_predictions = model(data.train_inputs).argmax(dim=1)
        validation_predictions = model(data.validation_inputs).argmax(dim=1)
        test_logits = model(data.test_inputs)
        test_predictions = test_logits.argmax(dim=1)

    train_accuracy = _accuracy(train_predictions, data.train_targets)
    validation_accuracy = _accuracy(validation_predictions, data.validation_targets)
    test_accuracy = _accuracy(test_predictions, data.test_targets)
    matrix = confusion_matrix(data.test_targets.numpy(), test_predictions.numpy())

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "train_min": data.train_min.clone(),
        "train_max": data.train_max.clone(),
        "feature_names": list(data.feature_names),
        "target_names": list(data.target_names),
        "input_size": INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "output_size": OUTPUT_SIZE,
        "seed": seed,
        "train_indices": list(data.train_indices),
        "validation_indices": list(data.validation_indices),
        "test_indices": list(data.test_indices),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)

    loaded = load_model_checkpoint(checkpoint_path)
    if loaded.model is model:
        raise AssertionError("checkpoint must be loaded into a new model instance")
    restored = restore_iris_data(loaded.payload)
    if not torch.equal(data.test_inputs, restored.test_inputs):
        raise AssertionError("checkpoint normalization did not reproduce test inputs exactly")
    with torch.no_grad():
        original_logits = model(data.test_inputs)
        loaded_logits = loaded.model(restored.test_inputs)
    if not torch.equal(original_logits, loaded_logits):
        raise AssertionError("original and loaded model logits differ")
    if not torch.equal(original_logits.argmax(dim=1), loaded_logits.argmax(dim=1)):
        raise AssertionError("original and loaded model classes differ")

    return TrainingResult(
        model=model,
        data=data,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        train_accuracy=train_accuracy,
        validation_accuracy=validation_accuracy,
        test_accuracy=test_accuracy,
        confusion=matrix,
        checkpoint_path=checkpoint_path,
    )


def _accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    return float((predictions == targets).float().mean().item())


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the reproducible Iris MLP.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()

    result = train_iris(args.checkpoint, epochs=args.epochs, seed=SEED)
    print(f"best epoch: {result.best_epoch}")
    print(f"best validation loss: {result.best_validation_loss:.6f}")
    print(f"train accuracy: {result.train_accuracy:.2%}")
    print(f"validation accuracy: {result.validation_accuracy:.2%}")
    print(f"test accuracy: {result.test_accuracy:.2%}")
    print("confusion matrix:")
    print(result.confusion)
    print("checkpoint reload: PASS (new instance, exact parameters/logits/classes)")
    print(f"checkpoint: {result.checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
