import numpy as np

class Trainer:
    def __init__(self, network, optimizer):
        self.network = network
        self.optimizer = optimizer

    def train_step(self, x_batch, t_batch):
        # Forward + Backward + Update (من Lab4/Lab5)
        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.parameters(), grads)

    def fit(self, x_train, t_train, x_test, t_test, epochs=20, batch_size=100, verbose=1):
        train_size = x_train.shape[0]
        iter_per_epoch = max(train_size / batch_size, 1)

        for i in range(epochs):
            for j in range(int(iter_per_epoch)):
                batch_mask = np.random.choice(train_size, batch_size)
                x_batch = x_train[batch_mask]
                t_batch = t_train[batch_mask]
                self.train_step(x_batch, t_batch)

            if verbose:
                train_loss = self.network.loss(x_train, t_train)
                train_acc = self.network.accuracy(x_train, t_train)
                test_acc = self.network.accuracy(x_test, t_test)
                print(f"Epoch {i+1}: loss={train_loss}, train_acc={train_acc}, test_acc={test_acc}")