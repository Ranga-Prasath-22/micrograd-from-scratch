# Micrograd from Scratch

A step-by-step implementation of a tiny autograd engine and neural network library, inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

## Overview

This project builds an automatic differentiation engine from scratch, demonstrating the core concepts behind modern deep learning frameworks like PyTorch.

## Progress

- [x] **Day 1**: `Value` class with basic operations (`+`, `*`) and computation graph visualization
- [x] **Day 2**: Manual backpropagation, gradient checking, and a simple neuron
- [x] **Day 3**: Chain rule, automatic `backward()`, topological sort, gradient accumulation

## Structure

```
├── day_01_value_class.ipynb       # Day 1: Value object and basic operations
├── day_02_backpropagation.ipynb   # Day 2: Backpropagation and neurons
├── day_03_chain_rule.ipynb        # Day 3: Automatic backward pass
└── README.md
```

## Getting Started

```bash
pip install graphviz matplotlib numpy
```

## Requirements

- Python 3.8+
- graphviz
- matplotlib
- numpy
