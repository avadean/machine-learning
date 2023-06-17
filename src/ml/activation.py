from numpy import ndarray, exp, maximum


def linear(z: ndarray):
    return z


def sigmoid(z: ndarray):
    return 1.0 / (1.0 + exp(-z))


def relu(z: ndarray):
    return maximum(z, z / 100.0)


def swish(z: ndarray, beta: float = 1.0):
    assert beta > 0.0
    return z * sigmoid(z * beta)
