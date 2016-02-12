from __future__ import absolute_import
from __future__ import print_function
import os
import time

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adadelta
from keras.utils import np_utils
from keras.layers.advanced_activations import PReLU


def main(job_id, params):
    print('spear_wrapper job #:%s' % str(job_id))
    print("spear_wrapper in directory: %s" % os.getcwd())
    print("spear_wrapper params are:%s" % params)


    # return run_cifar10(params)
    return run_mnist_small()

def run_mnist_small():
    timestring = str(int(time.time()))
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
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
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
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Flatten())


    model.add(Dense(512))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=adadelta)

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    timestring1 = str(int(time.time()))
    print("time used is "+ timestring1+"\t"+timestring)
    return score[0]


def write_params(input_file, output_file, params):
   # log("spear_wrapper params are:%s" % params);
    """
    go thorugh the hyperyaml line by line to create tempyaml
    """
    with open(input_file, 'r') as fin:
        with open(output_file, 'w') as fout:
            for line in fin:
                if '!hyperopt' in line:
                    line = parse_line(line, params)
                fout.write(line)


def parse_line(line, params):
    #log("spear_wrapper params are:%s" % params);
    """
    Replace the line defining the parameter range by just a name value pair.
    """
    dic = [k.strip("{},") for k in line.split()]
    out = params[dic[2]][0]
    return dic[0] + " " + str(out) + ",\n"
if __name__=='__main__':
    run_mnist_small()