import itertools as it
from copy import copy

import numpy as np
import numpy.random as npr

from funkyyak import grad, kylist
from hypergrad.nn_utils import BatchList, VectorParser
from hypergrad.optimizers import sgd, sgd2, sgd3, sgd_parsed
from hypergrad.optimizers import simple_sgd, rms_prop, adam

npr.seed(0)

def nd(f, *args):
    unary_f = lambda x : f(*x)
    return unary_nd(unary_f, args)

def unary_nd(f, x):
    eps = 1e-2
    if isinstance(x, np.ndarray):
        nd_grad = np.zeros(x.shape)
        for dims in it.product(*map(range, x.shape)):
            nd_grad[dims] = unary_nd(indexed_function(f, x, dims), x[dims])
        return nd_grad
    elif isinstance(x, tuple):
        return tuple([unary_nd(indexed_function(f, list(x), i), x[i])
                      for i in range(len(x))])
    elif isinstance(x, dict):
        return {k : unary_nd(indexed_function(f, x, k), v) for k, v in x.iteritems()}
    elif isinstance(x, list):
        return [unary_nd(indexed_function(f, x, i), v) for i, v in enumerate(x)]
    else:
        return (f(x + eps/2) - f(x - eps/2)) / eps

def indexed_function(fun, arg, index):
    local_arg = copy(arg)
    def partial_function(x):
        local_arg[index] = x
        return fun(local_arg)
    return partial_function


def make_optimization_problem(N_weights):
    true_argmin = npr.randn(N_weights)

    def loss_fun(x, i):
        """quadratic loss raised to a power"""
        return np.dot(x - true_argmin, x - true_argmin)**(1.1)
    return loss_fun, true_argmin

def test_simple_sgd():
    N_weights = 5
    W0 = 0.1 * npr.randn(N_weights)
    (loss_fun, true_argmin) = make_optimization_problem(N_weights)
    x_min = simple_sgd(grad(loss_fun), W0)
    assert np.allclose(x_min, true_argmin, rtol=1e-3, atol=1e-4), \
            "Diffs are: {0}".format(x_min - true_argmin)

def test_rms_prop():
    N_weights = 5
    W0 = 0.1 * npr.randn(N_weights)
    (loss_fun, true_argmin) = make_optimization_problem(N_weights)
    x_min = rms_prop(grad(loss_fun), W0)
    assert np.allclose(x_min, true_argmin, rtol=1e-3, atol=1e-4), \
            "Diffs are: {0}".format(x_min - true_argmin)

def test_adam():
    N_weights = 5
    W0 = 0.1 * npr.randn(N_weights)
    (loss_fun, true_argmin) = make_optimization_problem(N_weights)
    x_min = adam(grad(loss_fun), W0)
    assert np.allclose(x_min, true_argmin, rtol=1e-3, atol=1e-4), \
            "Diffs are: {0}".format(x_min - true_argmin)





def test_sgd():
    N_weights = 5
    W0 = 0.1 * npr.randn(N_weights)
    V0 = 0.1 * npr.randn(N_weights)
    N_data = 12
    batch_size = 4
    num_epochs = 3
    batch_idxs = BatchList(N_data, batch_size)
    N_iter = num_epochs * len(batch_idxs)
    alphas = 0.1 * npr.rand(len(batch_idxs) * num_epochs)
    betas = 0.5 + 0.2 * npr.rand(len(batch_idxs) * num_epochs)

    A = npr.randn(N_data, N_weights)
    def loss_fun(W, idxs):
        sub_A = A[idxs, :]
        return np.dot(np.dot(W, np.dot(sub_A.T, sub_A)), W)

    result = sgd(loss_fun, batch_idxs, N_iter, W0, V0, alphas, betas)
    d_x = result['d_x']
    d_v = result['d_v']
    d_alphas = result['d_alphas']
    d_betas = result['d_betas']

    def full_loss(W0, V0, alphas, betas):
        result = sgd(loss_fun, batch_idxs, N_iter, W0, V0, alphas, betas)
        x_final = result['x_final']
        return loss_fun(x_final, batch_idxs.all_idxs)

    d_an = (d_x, d_v, d_alphas, d_betas)
    d_num = nd(full_loss, W0, V0, alphas, betas)
    for i, (an, num) in enumerate(zip(d_an, d_num)):
        assert np.allclose(an, num, rtol=1e-3, atol=1e-4), \
            "Type {0}, diffs are: {1}".format(i, an - num)


def test_sgd2():
    N_weights = 5
    W0 = 0.1 * npr.randn(N_weights)
    V0 = 0.1 * npr.randn(N_weights)
    N_data = 12
    batch_size = 4
    num_epochs = 3
    batch_idxs = BatchList(N_data, batch_size)
    N_iter = num_epochs * len(batch_idxs)
    alphas = 0.1 * npr.rand(len(batch_idxs) * num_epochs)
    betas = 0.5 + 0.2 * npr.rand(len(batch_idxs) * num_epochs)
    meta = 0.1 * npr.randn(N_weights*2)

    A = npr.randn(N_data, N_weights)
    def loss_fun(W, meta, idxs):
        sub_A = A[idxs, :]
        return np.dot(np.dot(W + meta[:N_weights] + meta[N_weights:], np.dot(sub_A.T, sub_A)), W)

    def meta_loss_fun(w, meta):
        return np.dot(w, w) + np.dot(meta, meta)

    def full_loss(W0, V0, alphas, betas, meta):
        result = sgd2(loss_fun, meta_loss_fun, batch_idxs, N_iter, W0, V0, alphas, betas, meta)
        return result['L_final']

    def meta_loss(W0, V0, alphas, betas, meta):
        result = sgd2(loss_fun, meta_loss_fun, batch_idxs, N_iter, W0, V0, alphas, betas, meta)
        return result['M_final']

    result = sgd2(loss_fun, meta_loss_fun, batch_idxs, N_iter, W0, V0, alphas, betas, meta)

    d_an = (result['dLd_x'], result['dLd_v'], result['dLd_alphas'], result['dLd_betas'], result['dLd_meta'])
    d_num = nd(full_loss, W0, V0, alphas, betas, meta )
    for i, (an, num) in enumerate(zip(d_an, d_num)):
        assert np.allclose(an, num, rtol=1e-3, atol=1e-4), \
            "Type {0}, diffs are: {1}".format(i, an - num)
        print "Type {0}, diffs are: {1}".format(i, an - num)

    d_an = (result['dMd_x'], result['dMd_v'], result['dMd_alphas'], result['dMd_betas'], result['dMd_meta'])
    d_num = nd(meta_loss, W0, V0, alphas, betas, meta)
    for i, (an, num) in enumerate(zip(d_an, d_num)):
        assert np.allclose(an, num, rtol=1e-3, atol=1e-4), \
            "Type {0}, diffs are: {1}".format(i, an - num)
        print "Type {0}, diffs are: {1}".format(i, an - num)

def test_sgd3():
    N_weights = 5
    W0 = 0.1 * npr.randn(N_weights)
    V0 = 0.1 * npr.randn(N_weights)
    N_data = 12
    batch_size = 4
    num_epochs = 3
    batch_idxs = BatchList(N_data, batch_size)
    alphas = 0.1 * npr.rand(len(batch_idxs) * num_epochs)
    betas = 0.5 + 0.2 * npr.rand(len(batch_idxs) * num_epochs)
    meta = 0.1 * npr.randn(N_weights*2)

    A = npr.randn(N_data, N_weights)
    def loss_fun(W, meta, i=None):
        idxs = batch_idxs.all_idxs if i is None else batch_idxs[i % len(batch_idxs)]
        sub_A = A[idxs, :]
        return np.dot(np.dot(W + meta[:N_weights] + meta[N_weights:], np.dot(sub_A.T, sub_A)), W)

    def meta_loss_fun(w, meta):
        return np.dot(w, w) + np.dot(meta, meta)

    def full_loss(W0, V0, alphas, betas, meta):
        result = sgd3(loss_fun, meta_loss_fun, W0, V0, alphas, betas, meta)
        return loss_fun(result['x_final'], meta)

    def meta_loss(W0, V0, alphas, betas, meta):
        result = sgd3(loss_fun, meta_loss_fun, W0, V0, alphas, betas, meta)
        return meta_loss_fun(result['x_final'], meta)

    result = sgd3(loss_fun, meta_loss_fun, W0, V0, alphas, betas, meta)
    d_an = (result['dMd_x'], result['dMd_v'], result['dMd_alphas'],
            result['dMd_betas'], result['dMd_meta'])
    d_num = nd(meta_loss, W0, V0, alphas, betas, meta)
    for i, (an, num) in enumerate(zip(d_an, d_num)):
        assert np.allclose(an, num, rtol=1e-3, atol=1e-4), \
            "Type {0}, diffs are: {1}".format(i, an - num)

    result = sgd3(loss_fun, loss_fun, W0, V0, alphas, betas, meta)
    d_an = (result['dMd_x'], result['dMd_v'], result['dMd_alphas'],
            result['dMd_betas'], result['dMd_meta'])
    d_num = nd(full_loss, W0, V0, alphas, betas, meta )
    for i, (an, num) in enumerate(zip(d_an, d_num)):
        assert np.allclose(an, num, rtol=1e-3, atol=1e-4), \
            "Type {0}, diffs are: {1}".format(i, an - num)


def test_sgd_parser():
    N_weights = 6
    W0 = 0.1 * npr.randn(N_weights)
    N_data = 12
    batch_size = 4
    num_epochs = 4
    batch_idxs = BatchList(N_data, batch_size)

    parser = VectorParser()
    parser.add_shape('first',  [2,])
    parser.add_shape('second', [1,])
    parser.add_shape('third',  [3,])
    N_weight_types = 3

    alphas = 0.1 * npr.rand(len(batch_idxs) * num_epochs, N_weight_types)
    betas = 0.5 + 0.2 * npr.rand(len(batch_idxs) * num_epochs, N_weight_types)
    meta = 0.1 * npr.randn(N_weights*2)

    A = npr.randn(N_data, N_weights)
    def loss_fun(W, meta, i=None):
        idxs = batch_idxs.all_idxs if i is None else batch_idxs[i % len(batch_idxs)]
        sub_A = A[idxs, :]
        return np.dot(np.dot(W + meta[:N_weights] + meta[N_weights:], np.dot(sub_A.T, sub_A)), W)

    def full_loss(params):
        (W0, alphas, betas, meta) = params
        result = sgd_parsed(grad(loss_fun), kylist(W0, alphas, betas, meta), parser)
        return loss_fun(result, meta)

    d_num = nd(full_loss, (W0, alphas, betas, meta))
    d_an_fun = grad(full_loss)
    d_an = d_an_fun([W0, alphas, betas, meta])
    for i, (an, num) in enumerate(zip(d_an, d_num[0])):
        assert np.allclose(an, num, rtol=1e-3, atol=1e-4), \
            "Type {0}, diffs are: {1}".format(i, an - num)
