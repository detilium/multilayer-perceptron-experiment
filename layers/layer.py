# layer base class
#   all layers contain forward and backward propagation functions

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        """
        Forward propagation of the layer
        :param input: Input X to the layer
        :return: Output Y of the layer
        """
        raise NotImplementedError

    def backwards(self, output_error, learning_rate):
        """
        Backwards propagation of the layer, computing dE/dX and updating parameters if any
        :param output_error: Derivative of the error (dE/dY)
        :param learning_rate: ???
        :return: dE/dX
        """
        raise NotImplementedError
