import numpy as np

# Sigmoid

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))
