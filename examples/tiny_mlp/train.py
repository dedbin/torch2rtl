from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model import create_model


def make_dataset(samples: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(samples, 16)).astype(np.float32)
    block_sums = x.reshape(samples, 4, 4).sum(axis=2)
    y = block_sums.argmax(axis=1).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def train(epochs: int, output: Path, seed: int) -> None:
    set_seed(seed)
    x_train, y_train = make_dataset(4096, seed)
    model = create_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(x_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            preds = logits.argmax(dim=1)
            acc = (preds == y_train).float().mean().item()
            print(f"epoch={epoch:03d} loss={loss.item():.4f} acc={acc:.3f}")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    print(f"saved checkpoint to {output}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("tiny_mlp.pt"))
    args = parser.parse_args()
    train(args.epochs, args.out, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
