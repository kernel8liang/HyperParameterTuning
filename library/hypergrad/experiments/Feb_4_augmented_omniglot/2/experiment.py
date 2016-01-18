"""Runs for paper"""
import pickle

import numpy as np

import hypergrad as omniglot
from funkyyak import grad, getval
from hypergrad import RandomState, dictmap
from hypergrad import make_nn_funs, VectorParser
from hypergrad import random_partition
from hypergrad import sgd_meta_only as sgd

layer_sizes = [784, 400, 200, 55]
N_layers = len(layer_sizes) - 1
N_iters = 50
alpha = 1.0
beta = 0.9
seed = 0
N_thin = 10
N_scripts = 10
log_L2 = -4.0
log_init_scale = -2.0
N_meta_iter = 20
meta_alpha = 0.2
init_script_corr = 0.5

def run():
    RS = RandomState((seed, "top_rs"))
    all_data = omniglot.load_flipped_alphabets()
    train_data, tests_data = random_partition(all_data, RS, [12, 3])
    w_parser, pred_fun, loss_fun, frac_err = make_nn_funs(layer_sizes)
    N_weights = w_parser.vect.size
    script_parser = VectorParser()
    for i_script in range(N_scripts):
        script_parser[i_script] = np.zeros(N_weights)
    transform_parser = make_transform([0] * N_layers)

    def get_layers(vect):
        layers = []
        for i_layer in range(N_layers):
            weights_by_scripts = vect.reshape((N_scripts, N_weights))
            weights_idxs, _ = w_parser.idxs_and_shapes[('weights', i_layer)]
            biases_idxs, _  = w_parser.idxs_and_shapes[('biases',  i_layer)]
            assert weights_idxs.stop == biases_idxs.start
            layer_idxs = slice(weights_idxs.start, biases_idxs.stop)
            layers.append(weights_by_scripts[:, layer_idxs])
        return layers

    def transform_weights(z_vect, transform_vect):
        z_layers = get_layers(z_vect)
        transform = transform_parser.new_vect(transform_vect)
        w_layers = [np.dot(transform[i], z) for i, z in enumerate(z_layers)]
        return np.concatenate(w_layers, axis=1).ravel()

    def likelihood_loss(w_vect, data):
        w = script_parser.new_vect(w_vect)
        return sum([loss_fun(w[i], **script_data) for i, script_data in enumerate(data)])

    def regularization(z_vect):
        return np.dot(z_vect, z_vect) * np.exp(log_L2)

    def train_z(data, transform_vect, RS):
        def primal_loss(z_vect, transform_vect, i_primal, record_results=False):
            w_vect = transform_weights(z_vect, transform_vect)
            loss = likelihood_loss(w_vect, data)
            reg = regularization(z_vect)
            if record_results and i_primal % N_thin == 0:
                print "Iter {0}: train: {1}".format(i_primal, getval(loss) / N_scripts)
            return loss + reg
        z_vect_0 = RS.randn(script_parser.vect.size) * np.exp(log_init_scale)
        return sgd(grad(primal_loss), transform_vect, z_vect_0, alpha, beta, N_iters)

    def train_sharing():
        def hyperloss(transform_vect, i_hyper):
            RS = RandomState((seed, i_hyper, "hyperloss"))
            cur_train_data, cur_valid_data = random_partition(train_data, RS, [10, 2])
            z_vect_final = train_z(cur_train_data, transform_vect, RS)
            w_vect_final = transform_weights(z_vect_final, transform_vect)
            return likelihood_loss(w_vect_final, cur_valid_data) / N_scripts
        hypergrad = grad(hyperloss)
        cur_transform_vect = make_transform([init_script_corr] * N_layers).vect
        for i_hyper in range(N_meta_iter):
            print "Hyper iter {0}".format(i_hyper)
            grad_transform = hypergrad(cur_transform_vect, i_hyper)
            cur_transform_vect = cur_transform_vect - grad_transform * meta_alpha
        return cur_transform_vect

    transform_vects, train_losses, tests_losses = {}, {}, {}
    transform_vects['no_sharing']      = make_transform([0, 0, 0]).vect
    transform_vects['full_sharing']    = make_transform([1, 0, 0]).vect
    transform_vects['learned_sharing'] = train_sharing()
    for name in transform_vects.keys():
        RS = RandomState("final_training")
        tv = transform_vects[name]
        trained_z = train_z(train_data, tv, RS)
        trained_w = transform_weights(trained_z, tv)
        train_losses[name] = likelihood_loss(trained_w, train_data) / N_scripts
        tests_losses[name] = likelihood_loss(trained_w, tests_data) / N_scripts
        print "{0} : train: {1}, test: {2}".format(name, train_losses[name], tests_losses[name])
    return transform_parser, transform_vects, train_losses, tests_losses

def make_transform(layer_corr):
    diag = np.eye(N_scripts)
    full = np.full((N_scripts, N_scripts), 1.0 / N_scripts)
    transform_parser = VectorParser()
    for i_layer, corr in  enumerate(layer_corr):
        transform_parser[i_layer] = (1 - corr) * diag + corr * full
    return transform_parser

def T_to_covar(t):
    return np.dot(t, t.T)

def covar_to_corr(A):
    A_std = np.sqrt(np.diag(A))
    return (A / (A_std[:, None] * A_std[None, :]))

def build_covar_image(transform_vect, corr=True, use_abs=False):
    T = make_transform([0] * N_layers).new_vect(transform_vect).as_dict()
    covar = dictmap(T_to_covar, T)
    if use_abs:
        covar = dictmap(np.abs, covar)
    if corr:
        covar = dictmap(covar_to_corr, covar)
    return np.concatenate([covar[i] for i in range(N_layers)[::-1]], axis=0)

def plot():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'serif'

    with open('results.pkl') as f:
        transform_parser, transform_vects, train_losses, tests_losses = pickle.load(f)

    omniglot.show_alphabets(omniglot.load_flipped_alphabets(normalize=False))

    # Plotting transformations
    names = ['no_sharing', 'full_sharing', 'learned_sharing']
    title_strings = {'no_sharing'      : 'Independent nets',
                     'full_sharing'    : 'Shared bottom layer',
                     'learned_sharing' : 'Learned sharing'}
    covar_imgs = {name : build_covar_image(transform_vects[name]) for name in names}

    prop={'family':'serif', 'size':'12'}

    fig = plt.figure(0)
    fig.clf()
    fig.set_size_inches((6,6))
    for i, name in enumerate(names):
        ax = fig.add_subplot(1, 3, i + 1)
        ax.imshow(covar_imgs[name], cmap = mpl.cm.binary)
        ax.set_title(title_strings[name])
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            labels = ["Layer {0}".format(layer) for layer in [3, 2, 1]]            
            ypos   = [5, 15, 25]
            for s, y in zip(labels, ypos):
                ax.text(-2, y, s, rotation='vertical')
    plt.tight_layout()
    plt.savefig('learned_corr.png')
    plt.savefig('learned_corr.pdf')

if __name__ == '__main__':
    results = run()
    with open('results.pkl', 'w') as f:
        pickle.dump(results, f, 1)
    plot()
