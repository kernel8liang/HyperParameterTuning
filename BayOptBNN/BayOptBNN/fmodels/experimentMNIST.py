from __future__ import print_function
import numpy as np
from keras.optimizers import Adadelta
from keras.optimizers import SGD

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import sys
import os, os.path



class mnistExperiment:


    def __init__(self,input_dim, bounds=None, sd=None):
        self.input_dim = 4
        #learning-rate, momentum, dropout1, dropout2
        if bounds == None: self.bounds = [(0.001,4),(0.5,2),(0.2,0.7),(0.2,0.7)]
        else: self.bounds = bounds
        # self.min = [(0.0898,-0.7126),(-0.0898,0.7126)]
        # self.fmin = 0.005
        if sd==None: self.sd = 0
        else: self.sd=sd
        self.name = 'mnist '
        self.count=len(os.listdir(os.getcwd()))-1


    def f(self,params):
        # params = params.astype(np.float32)
        resultfinal = []
        size=params.shape[0]
        for i in range(0,size):
            resulttemp = []
            result = self.run_mnist_small(params[i])
            resulttemp.append(result)
            resultfinal.append(resulttemp)
        resultfinal = np.array(resultfinal)
        return resultfinal

    def run_mnist_small(self,params):

        # learning_rate, momentum, dropout1, dropout2 = params

        f = file("/home/jie/d3/fujie/hyper_parameter_tuning/BayOptBNN/experiment/BNN_MNIST/meta10/BO_BNN_mnist"+str(self.count)+".txt", 'w')
        orig_stdout = sys.stdout

        sys.stdout = f
        self.count = self.count+1
        #

        #
        learning_rate = 3.71853471
        momentum = 0.9
        dropout1 = 0.25
        dropout2 = 0.5

        batch_size = 128
        nb_classes = 10
        nb_epoch = 10

        # input image dimensions
        img_rows, img_cols = 28, 28
        # number of convolutional filters to use
        nb_filters = 32
        # size of pooling area for max pooling
        nb_pool = 2
        # convolution kernel size
        nb_conv = 3

        # the data, shuffled and split between tran and test sets
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)

        model = Sequential()

        model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                                border_mode='valid',
                                input_shape=(1, img_rows, img_cols)))
        model.add(Activation('relu'))
        model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(dropout1))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(dropout2))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        print('start compile')
        # adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)
        # model.compile(loss='categorical_crossentropy', optimizer=adadelta)
        decay=1e-6
        sgd = SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd)
        print('start fit')
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
        score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        sys.stdout = orig_stdout
        f.close()
        return score[0]







