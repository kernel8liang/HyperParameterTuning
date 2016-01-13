"""This example shows how to define the gradient of your own functions.
This can be useful for speed, numerical stability, or in cases where
your code depends on external library calls."""
from __future__ import absolute_import
from __future__ import print_function
import autogradwithbay.numpy as np
import autogradwithbay.numpy.random as npr

from autogradwithbay.core import grad, primitive
from autogradwithbay.util import quick_grad_check


# @primitive tells autogradwithbay not to look inside this function, but instead
# to treat it as a black box, whose gradient might be specified later.
# Functions with this decorator can contain anything that Python knows
# how to execute.
@primitive
def logsumexp(x):
    """Numerically stable log(sum(exp(x))), also defined in scipy.misc"""
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))

# Next, we write a function that specifies the gradient with a closure.
# The reason for the closure is so that the gradient can depend
# on both the input to the original function (x), and the output of the
# original function (ans).
def make_grad_logsumexp(ans, x):
    # If you want to be able to take higher-order derivatives, then all the
    # code inside this function must be itself differentiable by autogradwithbay.
    def gradient_product(g):
        # This closure multiplies g with the Jacobian of logsumexp (d_ans/d_x).
        # Because autogradwithbay uses reverse-mode differentiation, g contains
        # the gradient of the objective w.r.t. ans, the output of logsumexp.
        return np.full(x.shape, g) * np.exp(x - np.full(x.shape, ans))
    return gradient_product

# Now we tell autogradwithbay that logsumexmp has a gradient-making function.
logsumexp.defgrad(make_grad_logsumexp)


if __name__ == '__main__':
    # Now we can use logsumexp() inside a larger function that we want
    # to differentiate.
    def example_func(y):
        z = y**2
        lse = logsumexp(z)
        return np.sum(lse)

    grad_of_example = grad(example_func)
    print("Gradient: ", grad_of_example(npr.randn(10)))

    # Check the gradients numerically, just to be safe.
    quick_grad_check(example_func, npr.randn(10))
