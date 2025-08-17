import numpy as np
from keras.datasets import mnist
import keras.utils

from dense_layer import Dense
from convolutional import Convolutional
from reshape import Reshape
from sigmoid_activation import Sigmoid
from hyperbolic_tangent_activation import Tanh
from bce import binary_cross_entropy, binary_cross_entropy_prime

def preprocess_data(x, y, limit):
    zero_index = np.where(y==0)[0][:limit]
    one_index = np.where(y==1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") /255
    y = keras.utils.np_utils.to_categorical(y)
    y = y.reshape(len(y), 2, 1)
    return x, y

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

# CNN
network = [
    Convolutional((1,28,28), 3, 5),
    Sigmoid(),
    Reshape((5,26,26),(5*26*26, 1)),
    Dense(5*26*26, 100),
    Sigmoid(),
    Dense(100,2),
    Sigmoid()
]

epochs = 20
learning_rate = 0.1

# train
for e in range(epochs):
    error = 0
    for x, y in zip(x_train, y_train):
        # Forward
        output = x
        for layer in network:
            output = layer.forward(output)
        # error
        error += binary_cross_entropy(y, output)

        # backward propagation
        grad = binary_cross_entropy_prime(y, output)
        for layer in network:
            grad = layer.backward(grad, learning_rate)
    error /= len(x_train)
    print(f"{e + 1}/{epochs}, error={error}")

# test
for x, y in zip(x_test, y_test):
    output = x
    for layer in network:
        output = layer.forward(output)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")