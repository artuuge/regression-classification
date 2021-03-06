## regression-classification

The present repository is meant to provide some model examples of solving regression and classification problems using *neural networks* (NN).

### Linear

Multivariate linear regression can be perceived as a very simple case of the machine learning theory (ML) corresponding to a neural network consisting of a single linear layer. If we have *n* independent variables (features) and *k* dependent variables (targets), then this layer has *n* inputs and *k* outputs. The case *(k, n) = (1, 1)* corresponds to a real valued linear function of one real variable, and the network reduces to a single neuron with just one non-intercept input. The present example defines a straightforward implementation in [Haskell](https://www.haskell.org/platform/) 
```
ghc -O2 linear.hs -Wall
```
of multivariate linear regression in terms of ML illustrating it with *(k, n) = (4, 10)* and 10000 randomly generated training examples.

### Softmax

In this example we consider a simple neural network (NN) consisting of two layers: the *top layer* and the *hidden layer*. Each layer performs a linear transformation followed by a *softmax* function. One may perceive the construction mentioned as a natural two-layer generalization of *multinomial logistic regression*. Requiring more resources than the maximum entropy classifier, and certainly much more than the *naive Bayes classifier*, this kind of approach applies to a wider range of problems. We test the corresponding Haskell implementation
```
ghc -O2 softmax.hs -Wall
```
by training our NN to recognize the quadrant of a point on a coordinate plane.

### Multi-tanh

Here we test the support for multiple hidden layers in a neural network. The initialization and performance tuning of the *deep learning (DL)* neural networks is more tricky than of the simple ones. In particular, it is necessary to take into account the *small gradients* and *overfitting* problems, so the *learning rate* and the *regularization parameter* need to be adjusted properly. One should also pay attention to the *noise level* added to the weights during the initialization process to spontaneously break the symmetry. The current example 
```
ghc -O2 multi-tanh.hs -Wall
```
shows how this can be done for the quadrant recognition problem where we take a DL neural network with two hidden *Tanh* layers and a *SoftMax* top layer. 