Netz - Clojure Neural Network Library
=====================================

Description
-----------

Netz is a Clojure implementation of a [multilayer
perceptron](http://en.wikipedia.org/wiki/Multilayer_perceptron) (MLP), a type of
feedforward artificial neural network.  Netz provides functions for training and
running MLPs.  Training is accomplished via vectorized gradient descent
batch backpropagation.

A description of this implementation of backpropagation algorithm can be found
in docs/backpropagation.pdf.

Netz uses [Incanter](http://incanter.org/) for matrix operations.

*NOTE: This project is not production ready!*  Netz has had very little real
world testing and training convergence is still much slower than more
sophisticated implementations.  For production ready implementations, try:

* [FANN](http://leenissen.dk/fann/wp/) (C, Java, Ruby, Python and many more..)
* [Encog](http://www.heatonresearch.com/encog) (Java and CLR)

Usage
-----

```clojure
(ns your-namespace
  (:require [netz.core :as netz]))

(def examples [[[0 0] [1]]
               [[0 1] [0]]
               [[1 0] [0]]
               [[1 1] [1]]])

(def network (netz/train examples {:hidden-neurons [2]}))

(netz/run network [0 0]) ; => [0.9176]
(netz/run network [0 1]) ; => [0.0549]
(netz/run network [1 0]) ; => [0.0728]
(netz/run network [1 1]) ; => [0.9307]
```

Options
-------

*:hidden-neurons* - A vector containing the number of neurons in each hidden
layer.  Set to [2 2] for two hidden layers with two neurons each, or [] for no
hidden layers.  Setting this option is recommended.  Default: One hidden layer
with the same number of hidden neurons as inputs.

*:learning-rate* - The learning rate used while training.  See docs.  Default:
0.25.

*:learning-momentum* - The learning momentum used while training.  See docs.
Default: 0.

*:regularization-constant* - The regularization constant (lambda) used to
penalize large weights.  Default: 0.

*:callback* - A callback function.  If provided, Netz will call this function
after every epoch of training.  Returning false or nil from this callback will
cause training to stop.  See netz.core/report-callback for an example. Default:
netz.core/report-callback.

*:callback-resolution - An integer specifying how often the callback function is
invoked.  Default: 100.

*:max-epochs* - An integer specifying the maximum number of training epochs.
Default: 20,000.

*:desired-error* - A float specifying the desired training set mean squared
error (MSE) used while training.  Training will stop once the MSE drops below
the desired error.

*:calc-batch-error-in-parallel* - Calculate example batch errors in parallel for
each epoch.  Default: true.

License and Copyright
---------------------

Netz is distributed under the MIT License.  See LICENSE.

Copyright © 2012 Nick Ewing
