Netz - Clojure Neural Network Library
=====================================

Description
-----------

Netz is a Clojure implementation of a [multilayer
perceptron](http://en.wikipedia.org/wiki/Multilayer_perceptron) (MLP), a type of
feedforward artificial neural network.  Netz provides functions for training and
running MLPs.  Training is accomplished via gradient descent batch
[Rprop](http://en.wikipedia.org/wiki/Rprop) or
[standard backpropagation](http://en.wikipedia.org/wiki/Backpropagation).

Netz implements Rprop as described by Riedmiller in
[Rprop - Description and Implementation
Details](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.21.3428).  A
description of Netz's standard backpropagation algorithm can be found in
docs/backpropagation.pdf.

Netz uses [Incanter](http://incanter.org/) for matrix operations.

**NOTE: This project is not production ready!**  Netz has had very little real
world testing and training convergence is still slower than more sophisticated
implementations.  For production ready implementations, try:

* [FANN](http://leenissen.dk/fann/wp/) (C, Java, Ruby, Python and many more...)
* [Neuroph](http://neuroph.sourceforge.net/) (Java)
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

*:learning-algorithm* - The algorithm to use while training.  Choose either
:rprop for the Rprop algorithm or :bprop for standard back propagation.
Default: :rprop.

*:bprop-learning-rate* - The learning rate used while training with the standard
backpropagation algorithm.  Default: 0.25.

*:bprop-learning-momentum* - The learning momentum used while training with the
standard backpropagation algorithm.  Default: 0.

*:rprop-init-update* - Initial update value used for Rprop.  Default: 0.1.

*:rprop-update-min* - Minimum update value used for Rprop.  Default: 1e-6.

*:rprop-update-max* - Maximum update value used for Rprop.  Default: 50.0.

*:rprop-increase-factor* - Increase factor for Rprop.  Default: 1.2.

*:rprop-decrease-factor* - Decrease factor for Rprop.  Default: 0.5.

*:regularization-constant* - The regularization constant (lambda) used to
penalize large weights.  Default: 0.

*:callback* - A callback function.  If provided, Netz will call this function
after every epoch of training.  Returning false or nil from this callback will
cause training to stop.  See netz.core/report-callback for an example. Default:
netz.core/report-callback.

*:callback-resolution* - An integer specifying how often the callback function is
invoked.  Default: 100.

*:max-epochs* - An integer specifying the maximum number of training epochs.
Default: 20,000.

*:desired-error* - A float specifying the desired training set mean squared
error (MSE) used while training.  Training will stop once the MSE drops below
the desired error.

*:calc-batch-error-in-parallel* - Calculate example batch errors in parallel for
each epoch.  Default: true.

*:weight-initialization-method* - The weight initialization method.  Choose
either :random for randomly initialized weights or :nguyen-widrow to use the
[Nguyen-Widrow](http://www.stanford.edu/class/ee373b/nninitialization.pdf)
initialization method.  Default: :nguyen-widrow.

*:weight-initialization-range* - Randomly initialized weights will be between
[-:weight-initialization-range .. :weight-initialization-range].  Default: 0.5.

License and Copyright
---------------------

Netz is distributed under the MIT License.  See LICENSE.

Copyright © 2012 Nick Ewing
