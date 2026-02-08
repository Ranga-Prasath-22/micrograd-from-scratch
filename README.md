# Micrograd from Scratch

A step-by-step implementation of a tiny autograd engine and neural network library, inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

## Overview

This project builds an automatic differentiation engine from scratch, demonstrating the core concepts behind modern deep learning frameworks like PyTorch.

## Progress

- [x] **Day 1**: `Value` class with basic operations (`+`, `*`) and computation graph visualization
- [x] **Day 2**: Manual backpropagation, gradient checking, and a simple neuron
- [x] **Day 3**: Chain rule, automatic `backward()`, topological sort, gradient accumulation
- [x] **Day 4**: Power, division, negation, subtraction with backward passes
- [x] **Day 5**: `Neuron` class with weights, bias, and tanh activation
- [x] **Day 6**: `Layer` and `MLP` classes — organizing neurons into networks

## Structure

```
├── day_01_value_class.ipynb       # Day 1: Value object and basic operations
├── day_02_backpropagation.ipynb   # Day 2: Backpropagation and neurons
├── day_03_chain_rule.ipynb        # Day 3: Automatic backward pass
├── day_04_more_operations.ipynb   # Day 4: Power, division, negation, subtraction
├── day_05_neurons.ipynb           # Day 5: Neuron class with tanh activation
├── day_06_layers.ipynb            # Day 6: Layer and MLP classes
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
