import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss_layer = None
        self.params = {}  # dict للأوزان

    def add_layer(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss_layer):
        self.loss_layer = loss_layer

    def init_weight(self, input_size, hidden_sizes, output_size, weight_init_std=0.01):
        # تهيئة أوزان (من Lab3/Lab4)
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(1, len(layer_sizes)):
            self.params[f'W{i}'] = weight_init_std * np.random.randn(layer_sizes[i-1], layer_sizes[i])
            self.params[f'b{i}'] = np.zeros(layer_sizes[i])

    def predict(self, x, train_flg=False):
        for layer in self.layers:
            if "Dropout" in layer.__class__.__name__ or "BatchNorm" in layer.__class__.__name__:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.loss_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(x.shape[0])

    def parameters(self):
        return self.params

    def gradient(self, x, t):
        # Backprop (من Lab5)
        self.loss(x, t)
        dout = 1
        dout = self.loss_layer.backward(dout)
        layers = list(self.layers)
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for idx in range(1, len(self.layers) // 2 + 1):
            grads[f'W{idx}'] = self.layers[2*idx-2].dW  # افتراض Affine + Activation
            grads[f'b{idx}'] = self.layers[2*idx-2].db
        return grads