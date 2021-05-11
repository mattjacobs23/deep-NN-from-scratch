"""
We use an optimizer to adjust the parameters of our network based on the gradients
computed during backpropagation

Might want to add optimizer with momentum, RMSprop, etc. SGD simple to implement. 
"""
from deepNN.nn import NeuralNet

class Optimizer:
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent performs update step after every iteration
    Gradient gives us direction and magnitude in which the function increases the fastest
    Negative of that will be direction decreasing the fastest.
    """
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr
    
    def step(self, net: NeuralNet) -> None:
        for param, grad in net.params_and_grads():
            param -= self.lr * grad