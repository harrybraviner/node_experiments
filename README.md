NODE Experiments
================

# Purpose

I want to reproduce from scratch the results of this paper: [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366).

The authors make available a PyTorch implementation [here](https://github.com/rtqichen/torchdiffeq).
I'll be using Tensorflow.

The general pattern of this repo is that notebooks are for running experiments, and for short-lived prototypes.
The majority of the code should live in modules.

## MNIST ResNet experiments

ResNet6 is used as a baseline for Chen et al.'s comparison to neural ODEs.
Their code is in PyTorch, but I have attempted to reproduce the method as faithfully as possible in tensorflow.

The `BaseNetwork` class contains all of the methods that will be common accross different networks, e.g. tracking accuracy and loss, and holding the training step.
This is then sublassed by particular networks, e.g. `FCNet` and `ResNet6`.
