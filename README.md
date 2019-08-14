NODE Experiments
================

# Purpose

I want to reproduce from scratch the results of this paper: [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366).

The authors make available a PyTorch implementation [here](https://github.com/rtqichen/torchdiffeq).
I'll be using Tensorflow.

## MNIST ResNet experiments

ResNet6 is used as a baseline for Chen et al.'s comparison to neural ODEs.
Their code is in PyTorch, but I have attempted to reproduce the method as faithfully as possible in tensorflow.
