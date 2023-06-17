from ml.layer import Layer
from numpy import ndarray
from typing import Callable


class Model:
    def __init__(self,
                 layers: list[Layer]):
        assert isinstance(layers, list)

        self.layers = layers

        assert len(self.layers) == 1  # TODO: temporary
        assert self.layers[-1].units == 1  # TODO: temporary (?)

    def fit(self, x: ndarray[float], y: ndarray[float],
            loss: Callable, derivative: Callable,
            alpha: float = 0.5, epochs: int = 1000):
        """ Gradient descent """

        assert len(self.layers) == 1  # TODO: temporary

        layer = self.layers[-1]

        assert layer.units == 1  # TODO: temporary

        layer.gradientDescent(x=x, y=y,
                              loss=loss, derivative=derivative,
                              alpha=alpha, iterations=epochs)
