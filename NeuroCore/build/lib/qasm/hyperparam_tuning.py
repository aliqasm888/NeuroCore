import numpy as np
from .network import NeuralNetwork
from .trainer import Trainer

class HyperparameterTuning:
    def __init__(self, param_grid):
        self.param_grid = param_grid  # dict مثل {'lr': [0.01, 0.1], 'batch_size': [32, 64], ...}

    def grid_search(self, x_train, t_train, x_val, t_val):
        best_acc = -1
        best_params = {}

        # Grid search (من Lab6 pseudocode)
        for lr in self.param_grid.get('lr', [0.01]):
            for batch_size in self.param_grid.get('batch_size', [100]):
                # بناء شبكة جديدة (افتراض بسيط)
                net = NeuralNetwork()
                # أضف طبقات حسب param_grid إذا حدد (مثل hidden_sizes)
                # هنا افتراض: init_weight و add_layer يدوياً
                optimizer = self.param_grid['optimizer'](lr=lr)  # افتراض SGD إذا لم يحدد
                trainer = Trainer(net, optimizer)
                trainer.fit(x_train, t_train, x_val, t_val, epochs=10, batch_size=batch_size, verbose=0)
                acc = net.accuracy(x_val, t_val)
                if acc > best_acc:
                    best_acc = acc
                    best_params = {'lr': lr, 'batch_size': batch_size}
        return best_params, best_acc