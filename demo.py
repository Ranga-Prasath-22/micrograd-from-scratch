import random
from pathlib import Path

import matplotlib.pyplot as plt

from nn import MLP
from optim import Adam
from trainer import Trainer


def make_data(seed=2026, n=80):
    random.seed(seed)
    xs = []
    ys = []
    for _ in range(n):
        x1 = random.uniform(-2.0, 2.0)
        x2 = random.uniform(-2.0, 2.0)
        x3 = random.uniform(-2.0, 2.0)
        signal = 0.8 * x1 - 1.0 * x2 + 0.6 * x3 + 0.4 * x1 * x2
        ys.append(1.0 if signal > 0 else -1.0)
        xs.append([x1, x2, x3])
    return xs, ys


def split(xs, ys, ratio=0.8):
    n = int(len(xs) * ratio)
    return xs[:n], ys[:n], xs[n:], ys[n:]


def save_plot(history, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Val loss")
    plt.title("Micrograd Demo Training Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=140)
    plt.close()


def main():
    random.seed(42)
    xs, ys = make_data()
    train_xs, train_ys, val_xs, val_ys = split(xs, ys)

    model = MLP(3, [8, 8, 1])
    opt = Adam(model.parameters(), lr=0.01)
    trainer = Trainer(model, opt, batch_size=12)
    history = trainer.train(train_xs, train_ys, epochs=35, val_xs=val_xs, val_ys=val_ys)

    final_train = history["train_loss"][-1]
    final_val = history["val_loss"][-1]

    out_plot = Path("results/demo_training_curve.png")
    save_plot(history, out_plot)

    print("Demo complete.")
    print(f"Final train loss: {final_train:.6f}")
    print(f"Final val loss:   {final_val:.6f}")
    print(f"Saved plot:       {out_plot}")


if __name__ == "__main__":
    main()
