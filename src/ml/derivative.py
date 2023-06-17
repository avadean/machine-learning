from numpy import average, ndarray, zeros


def mse_d(x: ndarray[float], f: ndarray[float], y: ndarray[float]):
    """ Mean square error derivative w.r.t w """
    m, n = x.shape  # Number of data points, number of features.

    dj_dw = zeros(n)

    err = f - y

    for i in range(m):
        for j in range(n):
            dj_dw[j] += err[i] * x[i, j]

    dj_dw /= m
    dj_db = average(err)

    return dj_dw, dj_db  # (f - y) * x, f - y
