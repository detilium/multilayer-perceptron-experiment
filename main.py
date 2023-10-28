import numpy as np

from network import Network
from layers.fc_layer import FCLayer
from layers.activation_layer import ActivationLayer
from activation_functions.tanh import tanh, tanh_derivative
from activation_functions.sigmoid import sigmoid, sigmoid_derivative
from loss_functions.mse import mse, mse_prime

# training data
x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network()
net.add(FCLayer(2, 3))
net.add(ActivationLayer(tanh, tanh_derivative))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(tanh, tanh_derivative))


# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=10000, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)

# tanh -> tanh:         [array([[-0.00124085]]), array([[0.99507386]]), array([[0.9934471]]), array([[-0.0055815]])]
# tanh -> sigmoid:      [array([[0.00504041]]), array([[0.9854361]]), array([[0.9854615]]), array([[0.01689164]])]
# sigmoid -> tanh:      [array([[0.00030073]]), array([[0.98891859]]), array([[0.98834016]]), array([[0.00012717]])]
# sigmoid -> sigmoid:   [array([[0.02913214]]), array([[0.96044061]]), array([[0.97455697]]), array([[0.0427617]])]
