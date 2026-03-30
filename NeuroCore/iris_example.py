import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from qasm import NeuralNetwork, Trainer
from qasm.layers import Affine
from qasm.activations import ReLU
from qasm.losses import SoftmaxCrossEntropy
from qasm.optimizers import Adam

iris = load_iris()
X = iris.data          
y = iris.target         

T = np.eye(3)[y]       

X_train, X_test, T_train, T_test = train_test_split(
    X, T, test_size=0.2, random_state=42, stratify=y
)

print("حجم التدريب:", X_train.shape)   
print("حجم الاختبار :", X_test.shape)   
net = NeuralNetwork()


net.add_layer(Affine(
    W = np.random.randn(4, 64) * np.sqrt(2.0 / 4),    # He initialization
    b = np.zeros(64)
))
net.add_layer(ReLU())

net.add_layer(Affine(
    W = np.random.randn(64, 32) * np.sqrt(2.0 / 64),
    b = np.zeros(32)
))
net.add_layer(ReLU())

net.add_layer(Affine(
    W = np.random.randn(32, 3) * np.sqrt(2.0 / 32),
    b = np.zeros(3)
))

net.set_loss(SoftmaxCrossEntropy())

optimizer = Adam(lr=0.005)   
trainer = Trainer(net, optimizer)

print("\nبدء التدريب...\n")
trainer.fit(
    X_train, T_train,
    epochs=120,
    batch_size=16,
    verbose=1
)

train_acc = net.accuracy(X_train, T_train)
test_acc  = net.accuracy(X_test, T_test)

print(f"دقة التدريب (Training accuracy)  : {train_acc:.4f}")
print(f"دقة الاختبار  (Test accuracy)    : {test_acc:.4f}")
