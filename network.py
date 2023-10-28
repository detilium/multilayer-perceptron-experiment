class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_derivative = None

    def add(self, layer):
        """
        Add new layer to the network
        :param layer: The layer to add
        """
        self.layers.append(layer)

    def use(self, loss, loss_derivative):
        """
        Tell the network which loss function and
        :param loss:
        :param loss_derivative:
        :return:
        """
        self.loss = loss
        self.loss_derivative = loss_derivative

    def predict(self, input):
        # sample dimension first
        samples = len(input)
        result = []

        # run network over all samples
        for i in range(samples):
            output = input[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train, epochs, learning_rate):
        """
        Train the network based on ``x_train``, compared to the true ``y_train``
        :param x_train:Training dataset
        :param y_train:True dataset
        :param epochs:The total number of training loops
        :param learning_rate:How fast the network should learn (typically between ``0.1`` and ``0.00001``).
                This can also be defined as 'the aggressiveness of the steps we should take towards the global minimum
                using gradient descent'
        """
        #print('Epochs (training loops):', epochs)
        # sample dimension first
        samples = len(x_train)
        #print('x_train:', x_train)
        #print('y_train:', y_train)
        #print('Samples (len(x_train)):', samples)

        # training loop
        for i in range(epochs):
            # average loss (scalar) for this given iteration
            err = 0

            # loop through all training data samples 1-by-1
            # the data samples (x_train) is an array of data samples,
            # therefore we loop through each (x_train[j]), try to predict,
            # evaluate the prediction (output) in relation to the truth (y_train[j])
            # so that we can then make changes to the weights and biases in the different nodes and layers.
            # this is done as many times as we want, given by the epoch.
            for j in range(samples):
                # retrieve the training sample from the training data that will be run through the network
                output = x_train[j]
                #print('Output original:', output)

                # FORWARD PROPAGATION

                for layer in self.layers:
                    # forward propagate all nodes in the layer, with the current weights and biases.
                    # if this is the first sample, the weights and biases will be random,
                    # but if this is not the first sample, the weights and biases have been tweaked according
                    # to the loss function, which should get on closer to the truth
                    output = layer.forward(output)
                    #print('Output after layer {index}: {output}'.format(index=k, output=output))

                # COMPUTE LOSS

                # the error (vector) is simply the difference in the predicted output and the truth
                #   prediction - actual = error (vector)
                # in case of
                #   prediction = [-4, 3, 0, 7]
                #   actual = [0, 1, 2, 3]
                # the error (vector) will become
                #   [-4, 3, 0, 7] - [0, 1, 2, 3] = [-4, 2, 2, 4]
                #
                # example from the training data (x_train) = [1, 0]
                #   sample = [1, 0]
                #   truth = [0]
                #   ex. output (prediction) = [0.42985394]
                # following the above calculations to find the error (vector) this would be
                #   [0.42985394] - [0] = [-0.42985394]
                #error_vector = y_train[j] - output
                #print('Error (vector):', error_vector)

                # the loss is calculated using the loss function.
                # the loss functions is used to map the error (vector) to a single number (scalar), : loss (scalar).
                # the loss (scalar) can be referred to as the "goodness in prediction".
                # the closer this is to zero, the better the predictions are.
                # in this example MSE (Mean-squared error) is used.
                #loss = self.loss(y_train[j], output)

                # err is the sum of all loss (scalar) of this given training iterations' samples
                err += self.loss(y_train[j], output)
                #print('Error (scalar):', err)

                # BACKWARD PROPAGATION

                # this error (vector) is the derivative of the error (dE)
                # in respect to the output (dY) of the network in this sample
                # to further understand hos to calculate the derivative of the error (vector) (dE)
                # follow the code into the loss_prime function
                error = self.loss_derivative(y_train[j], output)
                #print('Derivative of error (vector) in respect to the output (dE/dY):', error)
                for layer in reversed(self.layers):
                    #print('Backwards propagate into layer ', k)
                    # traverse into the FCLayer class' backwards to follow the backwards propagation flow
                    error = layer.backwards(error, learning_rate)

            # as err is the sum of all loss (scalar) of all individual training samples in this given iteration,
            # if we want to calculate the average loss (scalar) for this iteration,
            # we need to divide the loss (scalar) by the number of samples.
            err /= samples
            print('epoch %d/%d  err=%f' % (i+1, epochs, err))
