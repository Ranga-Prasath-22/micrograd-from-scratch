import math


class Optimizer:
    def __init__(self, params, lr=0.01, weight_decay=0.0, clip_value=None):
        self.params = list(params)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.clip_value = clip_value

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0

    def _regularized_grad(self, p):
        g = p.grad
        if self.weight_decay:
            g += self.weight_decay * p.data
        if self.clip_value is not None:
            c = float(self.clip_value)
            if g > c:
                g = c
            elif g < -c:
                g = -c
        return g


class SGD(Optimizer):
    def step(self):
        for p in self.params:
            g = self._regularized_grad(p)
            p.data -= self.lr * g


class Momentum(Optimizer):
    def __init__(self, params, lr=0.01, beta=0.9, weight_decay=0.0, clip_value=None):
        super().__init__(params, lr=lr, weight_decay=weight_decay, clip_value=clip_value)
        self.beta = float(beta)
        self.v = [0.0 for _ in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            g = self._regularized_grad(p)
            self.v[i] = self.beta * self.v[i] + g
            p.data -= self.lr * self.v[i]


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
        clip_value=None,
    ):
        super().__init__(params, lr=lr, weight_decay=weight_decay, clip_value=clip_value)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.m = [0.0 for _ in self.params]
        self.v = [0.0 for _ in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        b1t = 1.0 - self.beta1**self.t
        b2t = 1.0 - self.beta2**self.t
        for i, p in enumerate(self.params):
            g = self._regularized_grad(p)
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g * g)
            m_hat = self.m[i] / b1t
            v_hat = self.v[i] / b2t
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
