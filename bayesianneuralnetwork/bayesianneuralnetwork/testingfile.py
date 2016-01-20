from __future__ import absolute_import
from __future__ import print_function

import matplotlib.pyplot as plt

import autogradwithbay.numpy as np
import autogradwithbay.numpy.random as npr
import autogradwithbay.scipy.stats.norm as norm

from autogradwithbay import grad
from autogradwithbay.examples.optimizers import adam
import sys
from bayesianneuralnetwork import bayesian_neural_network


def build_toy_dataset(n_data=40, noise_std=0.1):
    D = 1
    rs = npr.RandomState(0)
    inputs  = np.concatenate([np.linspace(0, 2, num=n_data/2),
                              np.linspace(6, 8, num=n_data/2)])
    targets = np.cos(inputs) + rs.randn(n_data) * noise_std
    inputs = (inputs - 4.0) / 4.0
    inputs  = inputs.reshape((len(inputs), D))
    targets = targets.reshape((len(targets), D))
    return inputs, targets

inputs, targets = build_toy_dataset()
model=bayesian_neural_network(inputs,targets, layer_sizes =[1, 10, 10, 1],L2_reg=0.01)