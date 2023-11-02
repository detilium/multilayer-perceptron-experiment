import numpy as np

from network import Network
from layers.fc_layer import FCLayer
from layers.activation_layer import ActivationLayer
from activation_functions.tanh import tanh, tanh_derivative
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
