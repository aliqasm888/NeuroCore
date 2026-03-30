import numpy as np
from ..layers.base import Layer

class Tanh(Layer):
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = np.tanh(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1 - self.out ** 2)
        return dx