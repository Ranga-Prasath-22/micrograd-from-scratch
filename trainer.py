import random


def mse_loss(preds, targets):
    n = len(preds)
    return sum((p - t) ** 2 for p, t in zip(preds, targets)) * (1.0 / n)


class Trainer:
    def __init__(self, model, optimizer, batch_size=4, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = int(batch_size)
        self.scheduler = scheduler

    def _batches(self, xs, ys):
        idx = list(range(len(xs)))
        random.shuffle(idx)
        for i in range(0, len(idx), self.batch_size):
            chunk = idx[i : i + self.batch_size]
            bx = [xs[j] for j in chunk]
            by = [ys[j] for j in chunk]
            yield bx, by

    def _dataset_loss(self, xs, ys):
        preds = [self.model(x) for x in xs]
        return mse_loss(preds, ys).data

    def train(self, xs, ys, epochs=20, val_xs=None, val_ys=None):
        history = {"train_loss": [], "val_loss": [], "lr": []}

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for bx, by in self._batches(xs, ys):
                preds = [self.model(x) for x in bx]
                loss = mse_loss(preds, by)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.data
                n_batches += 1

            train_loss = epoch_loss / max(1, n_batches)
            history["train_loss"].append(train_loss)

            if val_xs is not None and val_ys is not None:
                history["val_loss"].append(self._dataset_loss(val_xs, val_ys))
            else:
                history["val_loss"].append(None)

            history["lr"].append(self.optimizer.lr)

            if self.scheduler is not None:
                self.scheduler.step(epoch + 1)

        return history
