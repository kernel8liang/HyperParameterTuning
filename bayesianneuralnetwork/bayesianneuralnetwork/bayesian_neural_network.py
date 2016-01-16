import numpy as np
from bnn import BNN

class GPRegression(BNN):
    """
    Gaussian Process model for regression

    This is a thin wrapper around the models.GP class, with a set of sensible defaults

    :param X: input observations
    :param Y: observed values
    :param Norm normalizer: [False]


    .. Note:: Multiple independent outputs are allowed using columns of Y

    """

    def __init__(self, X, Y):

        super(BNN, self).__init__(X, Y)

