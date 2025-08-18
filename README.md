# CNN_Scratch

This project is a from-first-principles implementation of the core components of a Convolutional Neural Network (CNN) in pure Python with a solution of the MNIST problem using one-hot encoding with accuracty of around 99%.

It reproduces and explains — step-by-step — how convolution operations, padding, strides, kernel flipping, forward pass, and backward pass work internally.
To speed up development and focus on CNN logic, some base classes and helper functions were adapted from my earlier project: Neural_Network_Scratch.

📖 What This Project Covers


1️⃣ Convolution Operation

Sliding Window: Kernel moves over the input image in small steps.

Dot Product: For each position, multiply input patch by the kernel values and sum.

Output Size Formula (no padding, stride 1):
Y=I−K+1

2️⃣ Padding and Stride

Padding: Adds zeros around the image so kernels can scan borders.

Stride: Number of pixels the kernel jumps between applications.

Y = (I−K+2P​)/S +1

3️⃣ Forward Pass

Input image is convolved with one or more kernels.

Bias terms are added to each filter's output map.

Outputs are passed through activation functions (e.g., ReLU).

4️⃣ Backward Pass

Compute gradient of the loss with respect to the output of convolution.

Rotate kernel 180° to propagate error to previous layer.

Handle stride and padding effects in gradient computation.

Compute gradients for kernel weights and biases.

5️⃣ Activation Functions

ReLU: Rectified Linear Unit for non-linearity.

Sigmoid and Tanh can also be used (reused from previous project).

6️⃣ Loss Function

Mean Squared Error (MSE) used in this implementation for demonstration.

Easily replaceable with cross-entropy for classification tasks.

📂 Project Structure
cnn.py                           # CNN implementation (forward & backward)
conv_layer.py                    # Convolution layer class
activation.py                    # Activation functions
dense_layer.py                    # Dense (fully-connected) layer from previous project
mean_squared_error.py             # MSE loss function
utils.py                          # Helper functions
xor.py / mnist_example.py         # Example training scripts

🚀 Getting Started
Prerequisites

Python 3.8+

NumPy

Install & Run
git clone https://github.com/<your-username>/cnn-from-scratch.git
cd cnn-from-scratch
pip install numpy
python mnist_example.py

