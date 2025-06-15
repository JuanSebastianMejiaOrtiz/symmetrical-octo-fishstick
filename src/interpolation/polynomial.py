import numpy as np


# Phi_{j}(t) = ((t - c) / d)^{j-1}
def changeScale(t, tmin, tmax):
    c = (tmin + tmax) / 2
    d = (tmax - tmin) / 2
    return (t - c) / d


def polynomialInterpolation(x_values, y_values):
    '''
    Realiza interpolación polinomial de base monomial y
        devuelve el polinomio interpolador

    Parámetros:
    - x_values: array-like, puntos x conocidos.
    - y_values: array-like, valores y conocidos.

    Retorna:
    - coeff: coeficientes del polinomio interpolador
    - f: polinomio interpolador
    '''
    if len(x_values) != len(y_values):
        raise ValueError("t and y must have the same length")
    n = len(x_values)
    x_scaled = changeScale(x_values, min(x_values), max(x_values))
    A = np.vander(x_scaled, N=n, increasing=True)

    coeff = np.linalg.solve(A, y_values)

    def f(t):
        t = changeScale(t, min(x_values), max(x_values))
        return np.polyval(coeff[::-1], t)

    return coeff, f
