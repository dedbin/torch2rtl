from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data import CLASS_NAMES, DatasetConfig, generate_dataset
from model import DEFAULT_CHECKPOINT_PATH, ImageCNN


LOGGER = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("build/image_cnn/training_report.json"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--samples-per-class", type=int, default=80)
    parser.add_argument("--val-samples-per-class", type=int, default=32)
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_reproducible_seed(args.seed)

    train_cfg = DatasetConfig(
        samples_per_class=args.samples_per_class,
        noise=args.noise,
        seed=args.seed,
    )
    val_cfg = DatasetConfig(
        samples_per_class=args.val_samples_per_class,
        noise=args.noise,
        seed=args.seed + 1,
    )
    train_images, train_labels = generate_dataset(train_cfg)
    val_images, val_labels = generate_dataset(val_cfg)

    model = ImageCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model=model,
            images=train_images,
            labels=train_labels,
            optimizer=optimizer,
            loss_fn=loss_fn,
            seed=args.seed + epoch,
        )
        val_loss, val_acc = evaluate(model, val_images, val_labels, loss_fn)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )
        LOGGER.info(
            "epoch=%02d train_loss=%.4f train_acc=%.3f val_loss=%.4f val_acc=%.3f",
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": list(CLASS_NAMES),
            "train_config": train_cfg.__dict__,
            "val_config": val_cfg.__dict__,
            "history": history,
        },
        args.checkpoint,
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "class_names": list(CLASS_NAMES),
        "epochs": args.epochs,
        "final": history[-1],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("saved checkpoint -> %s", args.checkpoint)
    LOGGER.info("saved report -> %s", args.report)
    return 0


def set_reproducible_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(
    model: ImageCNN,
    images: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    seed: int,
) -> tuple[float, float]:
    model.train()
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(labels), generator=generator)
    total_loss = 0.0
    correct = 0

    for index in order.tolist():
        image = images[index]
        label = labels[index]
        optimizer.zero_grad(set_to_none=True)
        logits = model(image).unsqueeze(0)
        loss = loss_fn(logits, label.reshape(1))
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        correct += int(logits.argmax(dim=1).item() == int(label.item()))

    count = int(len(labels))
    return total_loss / count, correct / count


def evaluate(
    model: ImageCNN,
    images: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: nn.Module,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for image, label in zip(images, labels, strict=True):
            logits = model(image).unsqueeze(0)
            loss = loss_fn(logits, label.reshape(1))
            total_loss += float(loss.item())
            correct += int(logits.argmax(dim=1).item() == int(label.item()))

    count = int(len(labels))
    return total_loss / count, correct / count


if __name__ == "__main__":
    raise SystemExit(main())
