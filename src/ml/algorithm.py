from ml.activation import linear
from ml.derivative import mse_d
from ml.loss import mse
from ml.tools import cost, regression
from numpy import dot, ndarray
from typing import Callable
import math

def gradientDescent(x: ndarray[float], y: ndarray[float],
                    w: ndarray[float], b: float,
                    f_x: Callable = linear, loss: Callable = mse,
                    derivative: Callable = mse_d,
                    alpha: float = 0.5, iterations: int = 1000):
    """
    Performs the gradient descent algorithm to determine the best
    choice of w and b

    Args:
        x : features
        y : labels
        w : initial parameter values
        b : initial parameter value
        f_x : descriptor function
        loss : loss function
        derivative : derivatives of the loss function w.r.t w, b
        alpha : learning rate
        iterations : number of iterations
    """

    for k in range(1, iterations + 1):
        f = f_x(regression(x, w, b))

        dj_dw, dj_db = derivative(x, f, y)

        w -= alpha * dj_dw
        b -= alpha * dj_db

        cost_k = cost(loss(f, y))

        if k % 100 == 0:
            print(f'{k=:<7}', f'{cost_k=:8.4f}')

    return w, b
