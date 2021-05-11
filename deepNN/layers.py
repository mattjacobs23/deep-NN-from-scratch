"""
Our neural networks will be made up of layers
Each layer needs to pass its inputs forward and propagate gradients backward.
inputs -> Linear -> Tanh -> linear -> output

Could add sigmoid, ReLU, etc.
Could add LSTMs, Convolution layers
"""

from typing import Dict, Callable
import numpy as np

from deepNN.tensor import Tensor

class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    
    def forward(self, inputs: Tensor) -> Tensor:
        """
        Produce outputs corresponding to these inputs
        """
        raise NotImplementedError
    
    
    def backward(self, grad: Tensor) -> Tensor:
        """
        Backpropagate this gradient through the layer
        """
        raise NotImplementedError

class Linear(Layer):
    """
    Computes output = inputs @ w + b
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        # Inputs will be (batch_size, input_size)
        # Outputs will be (batch_size, output_size)

        # Super class constructor to get the Dict initialized
        super().__init__()

        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        outputs = inputs @ w + b
        """
        self.inputs = inputs 
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        """
        Gradient wrt output, backpropagate to get gradient wrt inputs of this
        linear layer, along the way compute gradient wrt params within this layer
        if y = f(x) and x = a * b + c
        then dy/da = f'(x) * b
        and dy/db = f'(x) * a
        and dy/dc = f'(x)

        if y = f(x) and x = a @ b + c
        then dy/da = f'(x) @ b.T  # where .T refers to Transpose
        and dy/db = a.T @ f'(x)
        and dy/dc = f'(x) # will need to sum across batch dimension
        """
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T

F = Callable[[Tensor], Tensor]

class Activation(Layer):
    """
    An activation layer just applies a non-linear function 
    elementwise to its inputs
    """
    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f 
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        """
        y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)

        f is the rest of neural net, g is this part being done by this layer
        g' is actually f_prime wrt inputs, f' is actually grad wrt outputs
        """
        return self.f_prime(self.inputs) * grad

def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x)
    return 1 - y ** 2

class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)