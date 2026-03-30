import numpy as np

class Trainer:
    def __init__(self, network, optimizer):
        self.network = network
        self.optimizer = optimizer

    def train_step(self, x_batch, t_batch):
        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)

    def fit(self, x_train, t_train, epochs=10, batch_size=32, verbose=1):
        train_size = x_train.shape[0]
        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(train_size)
            for i in range(0, train_size, batch_size):
                batch_idx = indices[i:i + batch_size]
                x_batch = x_train[batch_idx]
                t_batch = t_train[batch_idx]
                self.train_step(x_batch, t_batch)

            if verbose:
                loss = self.network.forward(x_train, t_train)
                acc = self.network.accuracy(x_train, t_train)
                print(f"Epoch {epoch} - loss: {loss:.4f} - acc: {acc:.4f}")