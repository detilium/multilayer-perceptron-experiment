import numpy as np

# Rectified linear unit

def relu(x):
    return np.max(0, x)


def relu_derivative(x):
    if x < 0:
        return 0
    return 1
