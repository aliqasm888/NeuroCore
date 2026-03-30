

import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.layers = []          
        self.loss_layer = None   
        self.params = {}         
        self.grads = {}           

    def add_layer(self, layer):
        self.layers.append(layer)
        if hasattr(layer, 'W') and hasattr(layer, 'b'):
            affine_count = sum(1 for l in self.layers if hasattr(l, 'W'))
            self.params[f'W{affine_count}'] = layer.W
            self.params[f'b{affine_count}'] = layer.b

    def set_loss(self, loss_layer):
        self.loss_layer = loss_layer

    def predict(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def forward(self, x, t):
        y = self.predict(x)
        return self.loss_layer.forward(y, t)

    def accuracy(self, x, t):
        y_pred = self.predict(x)
        y_argmax = np.argmax(y_pred, axis=1)
        if t.ndim != 1:
            t_argmax = np.argmax(t, axis=1)
        else:
            t_argmax = t
        return np.mean(y_argmax == t_argmax)

    def gradient(self, x, t):
        self.forward(x, t)
        dout = self.loss_layer.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        self.grads = {}
        affine_idx = 1
        for layer in self.layers:
            if hasattr(layer, 'dW') and hasattr(layer, 'db'):
                self.grads[f'W{affine_idx}'] = layer.dW
                self.grads[f'b{affine_idx}'] = layer.db
                affine_idx += 1
        return self.grads

    def train_step(self, x_batch, t_batch, learning_rate=0.1):
        grads = self.gradient(x_batch, t_batch)
        for key in self.params:
            self.params[key] -= learning_rate * grads[key]