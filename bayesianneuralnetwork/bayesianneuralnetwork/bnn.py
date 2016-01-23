from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import sys


import logging
import warnings
from GPy.util.normalizer import MeanNorm
logger = logging.getLogger("BNN")

import matplotlib.pyplot as plt

import autogradwithbay.numpy as np
import autogradwithbay.numpy.random as npr
import autogradwithbay.scipy.stats.norm as norm

from autogradwithbay.examples.black_box_svi import black_box_variational_inference
from autogradwithbay.examples.optimizers import adam


import matplotlib.pyplot as plt

import autogradwithbay.numpy as np
import autogradwithbay.numpy.random as npr
import autogradwithbay.scipy.stats.norm as norm

from autogradwithbay import grad
from autogradwithbay.examples.optimizers import adam
import sys


class BNN():
    """
    General purpose Bayesian neural network model

    :param X: input observations
    :param Y: output observations



    """

#     layer_sizes = [784, 300, 300, 10]
# N_layers = len(layer_sizes) - 1
# batch_size = 50
# N_iters = 5000
# N_train = 50000
# N_valid = 5000
# N_tests = 5000
#
# all_N_meta_iter = [0, 0, 20]
# alpha = 0.01
# meta_alpha = 0.2
# beta = 0.1
# seed = 0
# N_thin = 500
# N_meta_thin = 1
# log_L2 = -4.0
# log_init_scale = -3.0




    def __init__(self, X, Y,layer_sizes, L2_reg, model_optimize_restarts=1):

        self.X = X.copy()
        self.Y = Y
        self.layer_sizes = layer_sizes
        rbf = lambda x: norm.pdf(x, 0, 1)
        self.nonlinearity= rbf
        self.sq = lambda x: np.sin(x)
        noise_variance = 0.01
        self.num_weights, self.predictions, self.logprob = \
            self.make_nn_funs(layer_sizes, L2_reg, noise_variance ,self.nonlinearity )


        self.log_posterior = lambda weights, t: self.logprob(weights, X, Y)

        # Build variational objective.
        self.objective, self.gradient, self.unpack_params= \
            black_box_variational_inference(self.log_posterior, self.num_weights,
                                            num_samples=20)

        self.rs = npr.RandomState(0)
        init_mean    = self.rs.randn(self.num_weights)
        init_log_std = -5 * np.ones(self.num_weights)
        self.init_var_params = np.concatenate([init_mean, init_log_std])
        self.num_samples=20

        # variables used for running optimization process
        self.optimization_runs = []
        self.model_optimize_restarts=model_optimize_restarts
        self.verbosity= True





    def set_XY(self, X=None, Y=None):

        self.X = X
        self.Y = Y

    def set_X(self,X):
        """
        Set the input data of the model

        :param X: input observations
        :type X: np.ndarray
        """
        self.set_XY(X=X)

    def set_Y(self,Y):
        """
        Set the output data of the model

        :param X: output observations
        :type X: np.ndarray
        """
        self.set_XY(Y=Y)

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

    def unpack_layers(self,weights):
        shapes = zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        num_weight_sets = len(weights)
        for m, n in shapes:
            yield weights[:, :m*n]     .reshape((num_weight_sets, m, n)),\
                  weights[:, m*n:m*n+n].reshape((num_weight_sets, 1, n))
            weights = weights[:, (m+1)*n:]

    def make_nn_funs(self,layer_sizes, L2_reg, noise_variance, nonlinearity=np.tanh):
        """These functions implement a standard multi-layer perceptron,
        vectorized over both training examples and weight samples."""

        shapes = zip(layer_sizes[:-1], layer_sizes[1:])
        num_weights = sum((m+1)*n for m, n in shapes)



        def predictions(weights, inputs):
            """weights is shape (num_weight_samples x num_weights)
               inputs  is shape (num_datapoints x D)"""
            inputs = np.expand_dims(inputs, 0)
            for W, b in self.unpack_layers(weights):
                outputs = np.einsum('mnd,mdo->mno', inputs, W) + b
                inputs = nonlinearity(outputs)
            return outputs

        def logprob(weights, inputs, targets):
            log_prior = -L2_reg * np.sum(weights**2, axis=1)
            preds = predictions(weights, inputs)
            log_lik = -np.sum((preds - targets)**2, axis=1)[:, 0] / noise_variance
            return log_prior + log_lik

        return num_weights, predictions, logprob

    def callback(self,params, t, g):
        lower = self.objective(params, t)
        print("Iteration {} lower bound {}".format(t, lower))

    def optimize(self):
        print("Optimizing variational parameters...")
        self.init_var_params = adam(self.gradient, self.init_var_params,
                                  step_size=0.1, num_iters=1000, callback=self.callback)


    def optimize_restarts(self, num_restarts=10, robust=False, verbose=True):
        print("Optimizing variational parameters...")

        for i in range(num_restarts):
            self.init_var_params = adam(self.gradient, self.init_var_params,
                                  step_size=0.1, num_iters=1000, callback=self.callback)
            # try:
            #     if not parallel:
            #         if i>0: self.randomize()
            #         self.optimize(**kwargs)
            #     else:
            #         self.optimization_runs.append(jobs[i].get())
            #
            #     if verbose:
            #         print(("Optimization restart {0}/{1}, f = {2}".format(i + 1, num_restarts, self.optimization_runs[-1].f_opt)))
            # except Exception as e:
            #     if robust:
            #         print(("Warning - optimization restart {0}/{1} failed".format(i + 1, num_restarts)))
            #     else:
            #         raise e


    def predict(self, Xnew):
        """weights is shape (num_weight_samples x num_weights)
           inputs  is shape (num_datapoints x D)"""
        shapes = zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        num_weights = sum((m+1)*n for m, n in shapes)


        inputs = np.expand_dims(Xnew, 0)
        # inputs = Xnew
        mean, log_std = self.unpack_params(self.init_var_params)
        rs = npr.RandomState(0)
        samples = rs.randn(self.num_samples, self.num_weights) * np.exp(log_std) + mean
        for W, b in self.unpack_layers(samples):
            # dotvalue = np.dot(inputs, W)
            # outputs = dotvalue+ b
            outputs = np.einsum('mnd,mdo->mno', inputs, W) + b
            inputs = self.nonlinearity(outputs)

        mean = np.mean(outputs, axis=0)
        variance= outputs.var(axis=0)

        return mean,variance

    def copy(self):
        return self
