"""
FizzBuzz is the following problem:

For each of the numbers 1 to 100:
* if the number is divisable by 3 print "fizz"
* if the number is divisable by 5 print "buzz"
* if the number is divisable by 3 and 5 print "fizzbuzz"
* otherwise just print the number

Basically a 4-class classification problem
"""
from typing import List
import numpy as np

from deepNN.train import train
from deepNN.nn import NeuralNet
from deepNN.layers import Linear, Tanh
from deepNN.optimizer import SGD


def fizz_buzz_encode(x: int) -> List[int]:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 3 == 0:
        return [0, 0, 1, 0]
    elif x % 5 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]

"""
So we don't just memorize the results
Train on numbers bigger than 100 and try to predict on numbers less than 100
"""

def binary_encode(x: int) -> List[int]:
    """
    10 digit binary encoding of x
    For each number 0 to 9, right shift x by that many bits, & it with 1
    """
    return [x >> i & 1 for i in range(10)]

inputs = np.array([
    binary_encode(x)
    for x in range(101, 1024)
])

targets = np.array([
    fizz_buzz_encode(x)
    for x in range(101, 1024)    
])

net = NeuralNet([
    Linear(input_size=10, output_size=50), # output size of 50 works well and doesnt take forever to train
    Tanh(),
    Linear(input_size=50, output_size=4)
])

# default lr was too big, need to specify 0.001 instead (otherwise loss gets huge and overflows)
train(net,
      inputs,
      targets,
      num_epochs=5000,
      optimizer=SGD(lr = 0.001))

for x in range(1, 101):
    predicted = net.forward(binary_encode(x))
    # predicted will be array of size 4. Again cheating not using batches here but still works
    predicted_idx = np.argmax(predicted)
    actual_idx = np.argmax(fizz_buzz_encode(x))
    labels = [str(x), "fizz", "buzz", "fizzbuzz"]
    print(x, labels[predicted_idx], labels[actual_idx])