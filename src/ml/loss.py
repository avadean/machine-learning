from numpy import log, ndarray


def mse(f: ndarray[float], y: ndarray[float]):
    """ Mean square error """
    return (f - y) ** 2.0 / 2.0


def bce(f: ndarray[float], y: ndarray[float]):
    """ Binary cross entropy """
    return (y - 1.0) * log(1.0 - f) - y * log(f)
