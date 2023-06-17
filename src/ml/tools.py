from numpy import average, dot, ndarray


def cost(loss: ndarray[float]):
    return average(loss)


def regression(x: ndarray[float], w: ndarray[float], b: float):
    return dot(x, w) + b
