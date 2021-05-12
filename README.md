# deep-NN-from-scratch

Deep neural network library built from first principles using only numpy.

### Highly configurable, this library allows you to customize:

**Loss Functions:**
* Mean Squared Error (MSE)
* Cross Entopy (multi-class classification)

**Activation Functions:**
* Tanh
* Sigmoid
* Softmax
* ReLU

**Optimizer:**
* Stochastic Gradient Descent 
* Customizeable Mini-Batch GD *(Not yet implemented)*
* Adam *(Not yet implemented)*
* Adagrad *(Not yet implemented)*

You can choose however many layers you want, the number of epochs to train for, and the learning rate. As a functionality test and an example of how to use this library, I verified this library can be used to model the non-linear [XOR](https://github.com/mattjacobs23/deep-NN-from-scratch/blob/main/xor.py) function, and was able to achieve 92% accuracy on the [FizzBuzz](https://github.com/mattjacobs23/deep-NN-from-scratch/blob/main/fizzbuzz.py) problem.

### Future Improvements
* Implement regularization terms to loss functions to prevent overfitting
* Add LSTMs, convolution layers
* Add optimizers with momentum, adaptive learning rates
