# micrograd-from-scratch

Built from first principles: scalar autograd, MLP, optimizers, schedulers, and a tiny trainer loop.

This repo is both:
- a learning track (`day_01` to `day_13` notebooks)
- a runnable mini-framework (`nn.py`, `optim.py`, `trainer.py`, `scheduler.py`)

## What I Built

- Scalar autograd engine with computation graph and reverse-mode backprop
- Neural net stack: `Neuron` -> `Layer` -> `MLP`
- Optimizers: `SGD`, `Momentum`, `Adam`
- Training utilities: mini-batching, train/val tracking
- LR schedulers: `StepLR`, `ExponentialLR`, `CosineAnnealingLR`
- Benchmark artifact comparing optimizer convergence on same data/seed

## How It Works

1. Forward pass builds a graph of `Value` nodes.
2. `backward()` runs reverse-topological traversal and accumulates grads with `+=`.
3. Optimizer reads grads and updates parameter `.data`.
4. Trainer handles batches, history logging, and optional scheduler stepping.

## How To Run

```bash
python -m pip install matplotlib
```

Quick demo:

```bash
python demo.py
```

Benchmark (SGD vs Momentum vs Adam, same dataset/seed):

```bash
python benchmarks/compare_optimizers.py
```

## Benchmark Artifact

Running `python benchmarks/compare_optimizers.py` generates:

- `results/convergence_train.png`
- `results/convergence_val.png`
- `results/benchmark_summary.csv`
- `results/benchmark_summary.json`

The benchmark is fair by design:
- same dataset
- same model shape
- same random initialization seed
- only optimizer changes

## Design Tradeoffs

- Kept scalar-level autograd for clarity over speed.
- Chose MSE + tanh outputs for a minimal, transparent setup.
- Optimizers include only essential knobs to keep code readable.
- Not vectorized; this is a teaching-first build, not a throughput engine.

## Failure Cases / Limits

- Can be slow on bigger datasets because ops are scalar objects.
- Deep or long training can still be numerically brittle.
- No GPU support, no mixed precision, no checkpointing.
- `Value` is not a drop-in tensor replacement.

## Repo Layout

```text
day_01_value_class.ipynb
...
day_13_adam.ipynb
nn.py
optim.py
trainer.py
scheduler.py
benchmarks/compare_optimizers.py
demo.py
results/
```
