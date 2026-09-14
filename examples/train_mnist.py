"""Exit criterion for Phase 5 (plan.md): train on MNIST end-to-end from a
raw download to >97% test accuracy — with every batch fetched by the
Rust-side DataLoader workers.

Usage:
    python examples/train_mnist.py [--data-dir DATA] [--epochs N]

No external dataset libraries: MNIST IDX files are downloaded from the
official mirrors and parsed in pure Rust (`oxitorch.datasets.MNIST`).
"""

import argparse
import time

import numpy as np

import oxitorch as ox
from oxitorch import nn
from oxitorch.datasets import MNIST
from oxitorch.optim import AdamW
from oxitorch.utils.data import DataLoader


def build_model():
    """A 784-256-128-10 MLP with ReLU activations."""
    return nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def accuracy(model, images, labels, batch=512):
    correct = 0
    n = int(labels.shape[0])
    with ox.no_grad():
        for lo in range(0, n, batch):
            hi = min(lo + batch, n)
            imgs = images[lo:hi].reshape([hi - lo, 784])
            targets = labels[lo:hi]
            logits = model(imgs)
            pred = logits.argmax(1, False).reshape([-1])
            correct += float((pred == targets).sum().numpy())
    return correct / n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    t0 = time.time()
    train_ds = MNIST(args.data_dir, train=True, download=True)
    test_ds = MNIST(args.data_dir, train=False, download=True)
    print(
        f"MNIST ready in {time.time() - t0:.1f}s: "
        f"{len(train_ds)} train / {len(test_ds)} test "
        f"(parsed in Rust, batches fetched by {args.num_workers} Rust workers)"
    )

    model = build_model()
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ox.optim.CosineAnnealingLR(opt, T_max=args.epochs)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        seed=1234,
        drop_last=False,
        num_workers=args.num_workers,
    )

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            opt.zero_grad()
            x = images.reshape([-1, 784])  # (b, 1, 28, 28) -> (b, 784)
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits, labels.reshape([-1]).numpy().astype(int).tolist()
            )
            loss.backward()
            opt.step()
            total_loss += float(loss.numpy()) * x.shape[0]
            seen += x.shape[0]
        scheduler.step()

        test_acc = accuracy(model, *test_ds.tensors)
        print(
            f"epoch {epoch + 1:2d}/{args.epochs}: "
            f"loss {total_loss / seen:.4f}  test acc {test_acc * 100:.2f}%  "
            f"({time.time() - t0:.1f}s)"
        )

    final_acc = accuracy(model, *test_ds.tensors)
    print(f"\nfinal test accuracy: {final_acc * 100:.2f}%")
    if final_acc > 0.97:
        print("Phase 5 exit criterion MET (>97%).")
    else:
        print("Phase 5 exit criterion NOT met (<97%).")
    return 0 if final_acc > 0.97 else 1


if __name__ == "__main__":
    raise SystemExit(main())
