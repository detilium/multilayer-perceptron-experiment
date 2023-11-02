# Neural network experiment

First attempt at making a neural network to solve the XOR problem, using code from various resources, binding it all 
together in a network class, to then train and test the model.

This is what would be referred to as an MLP - a multilayer perceptron.

## Activation functions
The chosen activation function is `tanh`, though one might argue that `sigmoid` should be used in the last layer, to 
ensure a value between 0 and 1 whereas the `tanh` activation function does range between -1 and 1. In this example, 
`tanh` does provide a more correct prediction than `sigmoid`. 

## Loss function
The chosen loss function is `MSE` or `Mean-Squared Error`. This loss function is widely used in today's neural networks.

## Architecture

### Layer class
As all layers in the neural network should be able to forward and backward propagate, as well take input and produce a 
certain output, it makes sense to create a base `Layer` class.

From this basic `Layer` class, 2 types of layers are inheriting: `FCLayer` (_a fully connected layer_), and 
`ActivationLayer` (_activation functions_). One could argue that the activation functions are not a layer in themselves 
per se, but the implementation is simplified, as the activations functions are simply added to the network, as if it 
was a specific layer.

### Network class
The network class is what bind all layers and all nodes together. This class is the network itself. Instantiating this 
class is step 1 to creating our neural network. Once instantiated, we can add as many layers as we want, as well as 
defined the loss function we want to use. We can train the model using the function `fit()` and we can predict a given 
output for a given input, using the function `predict`.