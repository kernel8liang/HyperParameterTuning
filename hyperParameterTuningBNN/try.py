import GPy
import GPyOpt
from numpy.random import seed
seed(12345)

def myf(x):
    return (2*x)**2

bounds = [(-1,1)]

max_iter = 15

myProblem = GPyOpt.methods.BayesianOptimization(myf,bounds)
myProblem.run_optimization(max_iter)