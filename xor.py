"""
Cannot learn XOR with a linear function

0, 0 -> 0
1, 0 -> 1
0, 1 -> 1
1, 1 -> 0
"""
import numpy as np

from deepNN.train import train
from deepNN.nn import NeuralNet
from deepNN.layers import Linear, Tanh, Sigmoid, Softmax, ReLU

inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

targets = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

net = NeuralNet([
    Linear(input_size=2, output_size=2),
    Tanh(),
    Linear(input_size=2, output_size=2)
])

train(net, inputs, targets)

for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y)