from layers.layer import Layer


class ActivationLayer(Layer):
    def __init__(self, activation, activation_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output

    def backwards(self, output_error, learning_rate):
        #print('Backwards propagation into activation function')
        # activation layers have no weights or biases to update,
        # so we're simply going calculate the derivative of the error (vector) (dY) in relation to the input (dX)
        #   dE/dX
        # and return this to the previous layer.
        #
        # the derivative of the error (vector) (dE) within the activation layers, will always be the derivative of the
        # activation function.
        # for each activation function, we therefore need the corresponding formular of the derivative of that function.
        #
        # in this specific example, we're using tanh (np.tanh(x)) as our activation function
        # therefore we need the derivative of tanh (1-np.tanh(x)**2)
        return self.activation_derivative(self.input) * output_error
