from ml.algorithm import gradientDescent
from numpy import array, dot, ndarray
from numpy.random import random
from typing import Callable


class Layer:
    def __init__(self, units: int, activation: Callable):
        assert isinstance(units, int)
        assert isinstance(activation, Callable)

        self.units: int = units
        self.activation: Callable = activation


    def gradientDescent(self, x: ndarray[float], y: ndarray[float],
                        loss: Callable, derivative: Callable,
                        alpha: float = 0.5, iterations: int = 1000):

        assert self.units == 1  # TODO: temporary

        n = x.shape[1]  # Number of features.

        w_initial = random(n)
        b_initial = float(random(1))

        w, b = gradientDescent(x, y, w_initial, b_initial,
                               self.activation, loss, derivative,
                               alpha, iterations)

        print('w', w, type(w))
        print('b', b, type(b))
