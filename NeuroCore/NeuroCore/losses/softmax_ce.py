import numpy as np

def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=1, keepdims=True)
    else:
        x = x - x.max()
        return np.exp(x) / np.exp(x).sum()

class SoftmaxCrossEntropy:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = -np.sum(t * np.log(self.y + 1e-7)) / x.shape[0]
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx