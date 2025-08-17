import numpy as np
from neuralnetwork.layer import Layer

# we need it because the result of a convolutional layer is a 3d block and usually the
# last layer is a dense layer that operates on 2d blocks
class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)