
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
from qasm.hyperparam_tuning import HyperparameterTuning

digits = load_digits()
X = digits.data.astype(np.float32) / 16.0  
y = np.zeros((len(digits.target), 10))
y[np.arange(len(digits.target)), digits.target] = 1

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

tuner = HyperparameterTuning(X_train, y_train, X_val, y_val)

param_grid = {
    'learning_rate': [0.01, 0.1],
    'batch_size': [32, 64],
    'optimizer': ['SGD', 'Adam'],
    'hidden_size': [50, 100],
    'activation': ['ReLU', 'Sigmoid']
}

best_params, best_acc = tuner.grid_search(param_grid, epochs=20)