from ..layers.base import Layer

class Linear(Layer):
    def forward(self, x):
        return x

    def backward(self, dout):
        return dout