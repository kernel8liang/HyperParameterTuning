import itertools
import os
import pickle

import numpy as np

from hypergrad.util import dictslice


def datapath(fname):

    datadir = os.path.expanduser('/Users/yumengyin/Desktop/hyper_parameter_tuning/hypergrad/data/mnist')
    return os.path.join(datadir, fname)

#load data as dictionary
def load_data_as_dict(data,totalClassNum, subClassIndexList={}, normalize=True ):
    X_train, y_train, X_test, y_test = data


    SubClassNum = subClassIndexList.__len__()

    updateClassNum = totalClassNum

    if normalize:
        train_mean = np.mean(X_train, axis=0)
        X_train = X_train - train_mean
        X_test = X_test - train_mean


    if SubClassNum != 0:
        X_train,y_train = select_subclassdata(X_train,y_train,totalClassNum,SubClassNum,subClassIndexList,normalize=normalize)
        X_test,y_test = select_subclassdata(X_test,y_test,totalClassNum,SubClassNum,subClassIndexList,normalize=normalize)
        updateClassNum = SubClassNum


    one_hot = lambda x, K : np.array(x[:,None] == np.arange(K)[None, :], dtype=int)
    partial_flatten = lambda x : np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    Y_train = one_hot(y_train, updateClassNum)
    Y_test = one_hot(y_test, updateClassNum)


    X_train = partial_flatten(X_train) / 255.0
    X_test  = partial_flatten(X_test)  / 255.0

    N_data_train = X_train.shape[0]
    N_data_test = X_test.shape[0]

    partitions = []
    partitions.append({'X' : X_train, 'T' : Y_train})
    partitions.append({'X' : X_test, 'T' : Y_test})
    # partitions.append(N_data_train)
    # partitions.append(N_data_test)

    return partitions

def select_subclassdata(X, y,totalClassNum,SubClassNum, subClassIndexList,normalize=True):


    X= np.array(list(itertools.compress(X, [subClassIndexList.__contains__(c) for c in y])))
    y= np.array(list(itertools.compress(y, [subClassIndexList.__contains__(c) for c in y])))


    d = {}
    for i in xrange(SubClassNum):
        d.update({subClassIndexList[i]: (totalClassNum+i)})

    d1 = {}
    for i in xrange(SubClassNum):
        d1.update({(totalClassNum+i): i})

    for k, v in d.iteritems():
        np.place(y,y==k,v)
    for k, v in d1.iteritems():
        np.place(y,y==k,v)
    return X,y



def loadSubsetData(data, RS, subset_sizes, clientNum):
    N_rows = data['X'].shape[0]
    partitions = []

    countPre = 0
    for i in range (0,clientNum):

        Count = (i*subset_sizes)/N_rows
        if Count> countPre:
            data = dictslice(data, RS.permutation(N_rows))
            countPre = Count
        startNum = (i*subset_sizes)%N_rows
        if (startNum+subset_sizes) > N_rows:
            idxs = slice(startNum, N_rows)
            idxs1 = slice(0, (N_rows-startNum))
            subset = list(itertools.chain(*zip(dictslice(data, idxs), dictslice(data,idxs1))))
        else:
            idxs = slice(startNum, startNum + subset_sizes)
            subset = dictslice(data, idxs)
        partitions.append(subset)
    return partitions



def loadMnist():
    with open(datapath("mnist_data.pkl")) as f:
        data = pickle.load(f)
    return data

if __name__=="__main__":

    data = loadMnist()
    all_data = load_data_as_dict(data, 10, subClassIndexList=[1,2,3,4])
    all_data = load_data_as_dict(data, 10, subClassIndexList=[1,2,3,4])
