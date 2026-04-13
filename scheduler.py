import math


class StepLR:
    def __init__(self, optimizer, step_size, gamma=0.1):
        self.optimizer = optimizer
        self.step_size = int(step_size)
        self.gamma = float(gamma)

    def step(self, epoch):
        if self.step_size > 0 and epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma


class ExponentialLR:
    def __init__(self, optimizer, gamma=0.99):
        self.optimizer = optimizer
        self.gamma = float(gamma)

    def step(self, epoch):
        _ = epoch
        self.optimizer.lr *= self.gamma


class CosineAnnealingLR:
    def __init__(self, optimizer, T_max, eta_min=0.0):
        self.optimizer = optimizer
        self.T_max = max(1, int(T_max))
        self.eta_min = float(eta_min)
        self.base_lr = float(optimizer.lr)

    def step(self, epoch):
        cos_term = (1.0 + math.cos(math.pi * epoch / self.T_max)) / 2.0
        self.optimizer.lr = self.eta_min + (self.base_lr - self.eta_min) * cos_term
