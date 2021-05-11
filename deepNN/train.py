"""
Here's a function that can train a neural net
"""
from deepNN.tensor import Tensor
from deepNN.nn import NeuralNet
from deepNN.loss import Loss, MSE
from deepNN.optimizer import Optimizer, SGD
from deepNN.data import DataIterator, BatchIterator

def train(net: NeuralNet,
          inputs: Tensor,
          targets: Tensor,
          num_epochs: int = 5000,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = MSE(),
          optimizer: Optimizer = SGD()) -> None:
        
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        # Get batches out of our iterator
        for batch in iterator(inputs, targets):
            # Make predictions
            predicted = net.forward(batch.inputs)
            # Compute epoch loss
            epoch_loss += loss.loss(predicted, batch.targets)
            # Compute gradient, derivative wrt every one of our predictions
            grad = loss.grad(predicted, batch.targets)
            # Propagate gradient backwards through network
            net.backward(grad)
            # Optimizer to take step on NN, adjust the weights
            optimizer.step(net)

        # End of each epoch, print epoch and the epoch loss
        print(epoch, epoch_loss)