class Layer:
    def forward(self, x):
        raise NotImplementedError("يجب تنفيذ forward")

    def backward(self, dout):
        raise NotImplementedError("يجب تنفيذ backward")