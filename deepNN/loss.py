"""
A loss function measures how good our predictions are
We can use this to adjust the parameters of our network

Do MSE here, but if doing a lot of multi-class classification might want to do
cross-entropy, add in regularization terms (would have to change the API for that)
"""
import numpy as np

from deepNN.tensor import Tensor

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    # gradient with respect to the weights, tensor of partial derivatives w respect to each of the predictions
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError

# Mean Squared Error (will just be total squared error actually)
class MSE(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)

    # gradient with respect to the weights, tensor of partial derivatives w respect to each of the predictions
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)