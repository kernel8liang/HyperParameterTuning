from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autogradwithbay.numpy as np
import autogradwithbay.numpy.random as npr
import autogradwithbay.scipy.stats.norm as norm

# from autogradwithbay.examples.black_box_svi import black_box_variational_inference
from autogradwithbay.examples.optimizers import adam


import matplotlib.pyplot as plt

import autogradwithbay.numpy as np
import autogradwithbay.numpy.random as npr
import autogradwithbay.scipy.stats.multivariate_normal as mvn
import autogradwithbay.scipy.stats.norm as norm

from autogradwithbay import grad
from autogradwithbay.examples.optimizers import adam
import sys

def black_box_variational_inference(logprob, D, num_samples):
    """Implements http://arxiv.org/abs/1401.0118, and uses the
    local reparameterization trick from http://arxiv.org/abs/1506.02557"""
    # sys.stdout = Logger("experiment.txt")

    def unpack_params(params):
        # Variational dist is a diagonal Gaussian.
        mean, log_std = params[:D], params[D:]
        return mean, log_std

    def gaussian_entropy(log_std):
        return 0.5 * D * (1.0 + np.log(2*np.pi)) + np.sum(log_std)

    rs = npr.RandomState(0)
    def variational_objective(params, t):
        """Provides a stochastic estimate of the variational lower bound."""
        mean, log_std = unpack_params(params)
        samples = rs.randn(num_samples, D) * np.exp(log_std) + mean
        lower_bound = gaussian_entropy(log_std) + np.mean(logprob(samples, t))
        loss = np.mean(logprob(samples, t))
        print("loss is "+ str(loss))
        return -lower_bound

    gradient = grad(variational_objective)

    return variational_objective, gradient, unpack_params


def make_nn_funs(layer_sizes, L2_reg, noise_variance, nonlinearity=np.tanh):
    """These functions implement a standard multi-layer perceptron,
    vectorized over both training examples and weight samples."""
    shapes = zip(layer_sizes[:-1], layer_sizes[1:])
    num_weights = sum((m+1)*n for m, n in shapes)

    def unpack_layers(weights):
        num_weight_sets = len(weights)
        for m, n in shapes:
            yield weights[:, :m*n]     .reshape((num_weight_sets, m, n)),\
                  weights[:, m*n:m*n+n].reshape((num_weight_sets, 1, n))
            weights = weights[:, (m+1)*n:]

    def predictions(weights, inputs):
        """weights is shape (num_weight_samples x num_weights)
           inputs  is shape (num_datapoints x D)"""
        inputs = np.expand_dims(inputs, 0)
        for W, b in unpack_layers(weights):
            outputs = np.einsum('mnd,mdo->mno', inputs, W) + b
            inputs = nonlinearity(outputs)
        return outputs

    def logprob(weights, inputs, targets):
        log_prior = -L2_reg * np.sum(weights**2, axis=1)
        preds = predictions(weights, inputs)
        log_lik = -np.sum((preds - targets)**2, axis=1)[:, 0] / noise_variance
        return log_prior + log_lik

    return num_weights, predictions, logprob


def build_toy_dataset(n_data=40, noise_std=0.1):
    D = 1
    rs = npr.RandomState(0)
    inputs  = np.concatenate([np.linspace(0, 2, num=n_data/2),
                              np.linspace(6, 8, num=n_data/2)])
    inputs1  = np.concatenate([np.linspace(0, 2, num=n_data/2),
                              np.linspace(6, 8, num=n_data/2)])
    targets = np.cos(inputs) + rs.randn(n_data) * noise_std
    inputs = (inputs - 4.0) / 4.0
    inputs1 = (inputs1 - 4.0) / 4.0
    inputs  = inputs.reshape((len(inputs), D))
    inputs1  = inputs.reshape((len(inputs), D))
    imputfull =  np.concatenate((inputs , inputs1),axis=1)
    targets = targets.reshape((len(targets), D))
    return imputfull, targets



import sys
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)


if __name__ == '__main__':

    sys.stdout = Logger("experiment.txt")
    # Specify inference problem by its unnor

    # Specify inference problem by its unnormalized log-posterior.
    rbf = lambda x: norm.pdf(x, 0, 1)
    sq = lambda x: np.sin(x)
    num_weights, predictions, logprob = \
        make_nn_funs(layer_sizes=[2, 10, 10, 1], L2_reg=0.01,
                     noise_variance = 0.01, nonlinearity=rbf)

    inputs, targets = build_toy_dataset()
    log_posterior = lambda weights, t: logprob(weights, inputs, targets)

    # Build variational objective.
    objective, gradient, unpack_params= \
        black_box_variational_inference(log_posterior, num_weights,
                                        num_samples=20)

    # # Set up figure.
    # fig = plt.figure(figsize=(8,8), facecolor='white')
    # ax = fig.add_subplot(111, frameon=False)
    # plt.ion()
    # plt.show(block=False)


    def callback(params, t, g):
        lower = objective(params, t)
        print("Iteration {} lower bound {}".format(t, lower))

        # Sample functions from posterior.
        rs = npr.RandomState(0)
        mean, log_std = unpack_params(params)
        # #rs = npr.RandomState(0)
        # sample_weights = rs.randn(10, num_weights) * np.exp(log_std) + mean
        # plot_inputs = np.linspace(-8, 8, num=400)
        # outputs = predictions(sample_weights, np.expand_dims(plot_inputs, 1))
        #
        # # Plot data and functions.
        # plt.cla()
        # ax.plot(inputs.ravel(), targets.ravel(), 'bx')
        # ax.plot(plot_inputs, outputs[:, :, 0].T)
        # ax.set_ylim([-2, 3])
        # plt.draw()
        # plt.pause(1.0/60.0)



    # Initialize variational parameters
    rs = npr.RandomState(0)
    init_mean    = rs.randn(num_weights)
    init_log_std = -5 * np.ones(num_weights)
    init_var_params = np.concatenate([init_mean, init_log_std])

    print("Optimizing variational parameters...")
    variational_params = adam(gradient, init_var_params,
                              step_size=0.1, num_iters=1000, callback=callback)
    # plt.savefig("result.pdf")
