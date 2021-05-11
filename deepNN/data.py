"""
Feed inputs into our network in batches
So here are some tools for iterating over data in batches
"""
from typing import Iterator, NamedTuple

import numpy as np

from deepNN.tensor import Tensor

# Has two fields, inputs and targets
Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])

class DataIterator:
    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator:
        raise NotImplementedError

class BatchIterator(DataIterator):
    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator:
        """
        Will shuffle the batches but not within the batches.
        Find the start indices and shuffle them
        """
        starts = np.arange(0, len(inputs), self.batch_size)
        if self.shuffle:
            np.random.shuffle(starts)
        
        # Use these starts to create batches
        for start in starts:
            end = start + self.batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]
            yield Batch(batch_inputs, batch_targets)
            """
            yield is a keyword in Python that is used to return from a function 
            without destroying the states of its local variable and when the function 
            is called, the execution starts from the last yield statement. 
            Any function that contains a yield keyword is termed as generator
            """
