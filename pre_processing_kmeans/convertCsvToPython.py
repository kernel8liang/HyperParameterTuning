from __future__ import absolute_import
from keras.datasets.data_utils import get_file
import numpy as np
import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils, generic_utils

import sys
from six.moves import cPickle
from six.moves import range
import itertools
import numpy as np
from keras.datasets import cifar100

train_path="/home/jie/.keras/datasets/cifar10_kmeans/trainData"
test_path="/home/jie/.keras/datasets/cifar10_kmeans/testData"

def load_batch(fpath):
    df = pd.read_csv(fpath)
    labels = df.target
    df = df.drop('target',1)
    data = df
    return data, labels


path = "/home/jie/.keras/datasets/cifar_csv/"
fpath =  path+"trainData.csv"
data, labels = load_batch(fpath)
fpath =  path+"testData.csv"
data1, labels1  = load_batch(fpath)


with open(train_path, "wb") as f:
    cPickle.dump(data, f)
    cPickle.dump(labels, f)
with open(test_path, "wb") as f:
    cPickle.dump(data1, f)
    cPickle.dump(labels1, f)
print("end saving of the file ")

with open(train_path, "rb") as f:
    data_train = cPickle.load(f)
    label_train = cPickle.load(f)

