import csv
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nn import MLP
from optim import Adam, Momentum, SGD
from trainer import Trainer


def build_dataset(seed=1337, n=90):
    random.seed(seed)
    xs = []
    ys = []
    for _ in range(n):
        x1 = random.uniform(-2.0, 2.0)
        x2 = random.uniform(-2.0, 2.0)
        x3 = random.uniform(-2.0, 2.0)
        # Non-linear target: plane + interaction + curvature.
        signal = 0.9 * x1 - 1.1 * x2 + 0.5 * x3 + 0.7 * x1 * x2 - 0.2 * x3 * x3
        y = 1.0 if signal > 0 else -1.0
        xs.append([x1, x2, x3])
        ys.append(y)
    return xs, ys


def split_dataset(xs, ys, train_ratio=0.8):
    n_train = int(len(xs) * train_ratio)
    return xs[:n_train], ys[:n_train], xs[n_train:], ys[n_train:]


def run_one(name, epochs, seed, train_xs, train_ys, val_xs, val_ys):
    random.seed(seed)
    model = MLP(3, [8, 8, 1])
    params = model.parameters()

    if name == "sgd":
        opt = SGD(params, lr=0.03)
    elif name == "momentum":
        opt = Momentum(params, lr=0.02, beta=0.9)
    elif name == "adam":
        opt = Adam(params, lr=0.01)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

    trainer = Trainer(model, opt, batch_size=12)
    history = trainer.train(
        train_xs,
        train_ys,
        epochs=epochs,
        val_xs=val_xs,
        val_ys=val_ys,
    )
    return history


def summarize(name, history):
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    best_val = min(val_loss)
    best_epoch = val_loss.index(best_val) + 1
    return {
        "optimizer": name,
        "final_train_loss": train_loss[-1],
        "final_val_loss": val_loss[-1],
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
    }


def plot_curves(histories, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(next(iter(histories.values()))["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    for name, h in histories.items():
        plt.plot(epochs, h["train_loss"], label=f"{name.upper()} train")
    plt.title("Optimizer Convergence (Train Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "convergence_train.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, h in histories.items():
        plt.plot(epochs, h["val_loss"], label=f"{name.upper()} val")
    plt.title("Optimizer Convergence (Validation Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "convergence_val.png", dpi=140)
    plt.close()


def write_results(output_dir, summaries, histories):
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "benchmark_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    with (output_dir / "benchmark_curves.json").open("w", encoding="utf-8") as f:
        json.dump(histories, f, indent=2)

    with (output_dir / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["optimizer", "final_train_loss", "final_val_loss", "best_val_loss", "best_epoch"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)


def print_table(summaries):
    print("\nOptimizer Benchmark (same dataset + same init seed)")
    print("-" * 74)
    print(f"{'Optimizer':<12}{'Final Train':>14}{'Final Val':>14}{'Best Val':>14}{'Best Epoch':>12}")
    print("-" * 74)
    for row in summaries:
        print(
            f"{row['optimizer']:<12}"
            f"{row['final_train_loss']:>14.6f}"
            f"{row['final_val_loss']:>14.6f}"
            f"{row['best_val_loss']:>14.6f}"
            f"{row['best_epoch']:>12}"
        )
    print("-" * 74)


def main():
    epochs = 40
    run_seed = 42
    data_seed = 1337

    xs, ys = build_dataset(seed=data_seed, n=90)
    train_xs, train_ys, val_xs, val_ys = split_dataset(xs, ys, train_ratio=0.8)

    histories = {}
    for name in ["sgd", "momentum", "adam"]:
        histories[name] = run_one(
            name=name,
            epochs=epochs,
            seed=run_seed,
            train_xs=train_xs,
            train_ys=train_ys,
            val_xs=val_xs,
            val_ys=val_ys,
        )

    summaries = [summarize(name, histories[name]) for name in ["sgd", "momentum", "adam"]]

    output_dir = Path("results")
    write_results(output_dir, summaries, histories)
    plot_curves(histories, output_dir)
    print_table(summaries)
    print("\nSaved:")
    print(f"  - {output_dir / 'convergence_train.png'}")
    print(f"  - {output_dir / 'convergence_val.png'}")
    print(f"  - {output_dir / 'benchmark_summary.csv'}")
    print(f"  - {output_dir / 'benchmark_summary.json'}")


if __name__ == "__main__":
    main()
