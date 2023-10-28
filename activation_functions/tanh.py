import numpy as np

# Hyberbolic tangent

def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1-np.tanh(x)**2
