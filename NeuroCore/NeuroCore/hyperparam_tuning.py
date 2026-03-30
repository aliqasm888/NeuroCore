
import numpy as np
from .network import NeuralNetwork
from .trainer import Trainer
from .optimizers import SGD, Momentum, AdaGrad, Adam
from .activations.relu import ReLU
from .activations.sigmoid import Sigmoid
from .layers.affine import Affine
from .losses.softmax_ce import SoftmaxCrossEntropy

class HyperparameterTuning:
    def __init__(self, x_train, t_train, x_val, t_val):
        self.x_train = x_train
        self.t_train = t_train
        self.x_val = x_val
        self.t_val = t_val

    def grid_search(self, param_grid, epochs=20):
        best_acc = 0.0
        best_params = None
        results = []


        for lr in param_grid.get('learning_rate', [0.01]):
            for batch_size in param_grid.get('batch_size', [32]):
                for opt_name in param_grid.get('optimizer', ['SGD']):
                    for hidden_size in param_grid.get('hidden_size', [50]):
                        for act_name in param_grid.get('activation', ['ReLU']):
                            print(f"تجربة: lr={lr}, batch_size={batch_size}, optimizer={opt_name}, "
                                  f"hidden_size={hidden_size}, activation={act_name}")

                            net = NeuralNetwork()
                            input_size = self.x_train.shape[1]
                            output_size = self.t_train.shape[1]

                            np.random.seed(42)
                            W1 = 0.01 * np.random.randn(input_size, hidden_size)
                            b1 = np.zeros(hidden_size)
                            W2 = 0.01 * np.random.randn(hidden_size, output_size)
                            b2 = np.zeros(output_size)

                            net.add_layer(Affine(W1, b1))

                            if act_name == 'ReLU':
                                net.add_layer(ReLU())
                            elif act_name == 'Sigmoid':
                                net.add_layer(Sigmoid())
                            else:
                                from .activations.tanh import Tanh
                                net.add_layer(Tanh())

                            net.add_layer(Affine(W2, b2))
                            net.set_loss(SoftmaxCrossEntropy())

                            if opt_name == 'SGD':
                                optimizer = SGD(lr=lr)
                            elif opt_name == 'Momentum':
                                optimizer = Momentum(lr=lr)
                            elif opt_name == 'AdaGrad':
                                optimizer = AdaGrad(lr=lr)
                            else:
                                optimizer = Adam(lr=lr)

                            trainer = Trainer(net, optimizer)
                            trainer.fit(self.x_train, self.t_train, epochs=epochs, batch_size=batch_size, verbose=0)

                            val_acc = net.accuracy(self.x_val, self.t_val)
                            print(f"دقة التحقق: {val_acc:.4f}\n")

                            if val_acc > best_acc:
                                best_acc = val_acc
                                best_params = {
                                    'learning_rate': lr,
                                    'batch_size': batch_size,
                                    'optimizer': opt_name,
                                    'hidden_size': hidden_size,
                                    'activation': act_name
                                }

                            results.append((val_acc, best_params.copy() if best_params else None))

        print(" انتهى البحث ")
        print("أفضل إعدادات:")
        print(best_params)
        print(f"أفضل دقة على بيانات التحقق: {best_acc:.4f}")

        return best_params, best_acc