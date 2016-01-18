import itertools
import os
import pickle

import numpy as np

from hypergrad import dictslice


def datapath(fname):

    project_dir = os.environ['EXPERI_PROJECT_PATH']
    datadir = project_dir+"/library/hypergrad/data/mnist"
    # datadir = os.path.expanduser('/Users/yumengyin/Desktop/hyper_parameter_tuning/library/hypergrad/data/mnist')
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

#
train_path_1="/home/jie/.keras/datasets/cifar10_kmeans_numpy/trainData"
test_path_1="/home/jie/.keras/datasets/cifar10_kmeans_numpy/testData"
train_path="/home/jie/.keras/datasets/cifar10_kmeans_numpy/trainData.npz"
test_path="/home/jie/.keras/datasets/cifar10_kmeans_numpy/testData.npz"

# train_path_2="/home/jie/.keras/datasets/cifar10_kmeans_numpy/trainData1"

def loadCifar10():
    # with open(train_path, "rb") as f:
    #     data_train = cPickle.load(f)
    #     label_train = cPickle.load(f)
    # with open(test_path, "rb") as f:
    #     data_test = cPickle.load(f)
    #     label_test = cPickle.load(f)
    npzfile = np.load(train_path)
    data_train=npzfile['X']
    label_train=npzfile['y']
    npzfile = np.load(test_path)
    data_test=npzfile['X']
    label_test=npzfile['y']


    return data_train, label_train, data_test, label_test


if __name__=="__main__":



    data = loadCifar10()
    data_train, label_train, data_test, label_test = data
    # print("complete loading the file ")
    # d = {}
    # for i in xrange(11):
    #     d.update({i:i-1})
    #
    #
    # for k, v in d.iteritems():
    #     np.place(label_train,label_train==k,v)
    #     np.place(label_test,label_test==k,v)
    # outfile1 = TemporaryFile(train_path_1)
    # outfile2 = TemporaryFile(test_path_1)
    # np.savez(outfile1, X=data_train,y=label_train)
    # np.savez(outfile2, X=data_test,y=label_test)
    # print("complete saving the file ")


    # with open(train_path, "wb") as f:
    #     cPickle.dump(data_train, f)
    #     cPickle.dump(label_train, f)
    # with open(test_path, "wb") as f:
    #     cPickle.dump(data_test, f)
    #     cPickle.dump(label_test, f)
    # print("end saving of the file ")
    #
    all_data = load_data_as_dict(data, 10, subClassIndexList=[1,2,3,4])
    from hypergrad import RandomState
    RS = RandomState((0, "to p_rs"))
    all_data=  loadSubsetData(data, RS, 2000, 10)
    all_data = load_data_as_dict(data, 10, subClassIndexList=[1,2,3,4])