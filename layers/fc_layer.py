from layers.layer import Layer
import numpy as np

# Fully connected layer
#   all node outputs (x, scalar) from the previous layer, serves as a collection of inputs (X, vector) to an FC layer

class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backwards(self, output_error, learning_rate):
        """
        Backward propagation to update weights and biases
        in relation to the derivative error (vector) (``dE``) of this layer's output (``dY``): (``dE/dY``)
        :param output_error:The derivative of the error (vector) of this layer's output (``dE/dY``)
        :param learning_rate:Definition of the aggressiveness that we should approach the local minimum of this layer
        :return:The derivative of the error (vector) (``dE``) in relation to the input (``dX``): ``dE/dX``
        """
        #print('Backwards output_error:', output_error)

        #print('Weights: {weights} - shape: {shape}'
              #.format(weights=self.weights, shape=self.weights.shape))

        #print('WeightsT (transformed): {weights} - shape: {shape}'
              #.format(weights=self.weights.T, shape=self.weights.T.shape))

        # calculate the derivative of the error (vector) (dE) in relation to the input (dX)
        #   dE/dX
        # the input error is used to back propagate to the layer before this layer, as the input of this layer (y)
        # is the output of the previous layer (x), as is therefore not used to update any variables within this
        # current layer, but we need to calculate dE/dX before we update the weights of this layer.
        #
        # the derivative of the error (vector) (dE) in relation to the input (dX) (input error) will become
        # the derivative of the error (vector) (dE) in relation to the output (dY) of the previous layer (output error)
        #   dE/dY
        # as the previous layer's output is this layer's input
        #
        # why do we transpose the weights?
        input_error = np.dot(output_error, self.weights.T)
        #print('Input error (dE/dX):', input_error)


        # calculate the derivative of the error (vector) (dE) in relation to the weights (dW)
        #   dE/dW
        # the weights error is used to update the weights of this layer's nodes by a given learning rate
        #
        # why do we transpose the input?
        #print('InputT (transformed):', self.input.T)
        weights_error = np.dot(self.input.T, output_error)
        #print('Weights error (dE/dW):', weights_error)

        # update weights accordingly to the derivative of the error (vector) (dE)
        # in relation to the weights (dW): dE/dW, in the negative direction of the derivative (hence the -)
        # multiplied by the learning rate, defining how aggressively we should alter the weights
        self.weights -= learning_rate * weights_error

        # update the biases accordingly to the derivative of the error (vector) (dE)
        # in relation to the output error (dY): dE/dY, in the negative direction of the derivative (hence the -)
        # multiplied by the learning rate, defining how aggressively we should alter the biases.
        #
        # the biases will always be updated directly from the derivative of the error (vector) (dE), in relation
        # to the output (dY): dE/dY, since we're adding (+) the biases to the output result (Y).
        #   self.output = np.dot(self.input, self.weights) *** + self.bias ***
        #   Y = X . W + B
        # since this is the case, there's a direct relation between the output (Y) and the bias
        self.bias -= learning_rate * output_error

        # we're returning the derivative of the error (vector) (dE) in relation to the input (dX)
        #   dE/dX
        # as the previous layer needs to use this as it's output error: dE/dY
        return input_error
