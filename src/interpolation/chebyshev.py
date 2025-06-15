import numpy as np
from polynomial import polynomialInterpolation


def chebyshev_interpolation(x_values, y_values, n=None):
    """
    Realiza interpolación usando nodos de Chebyshev y
        devuelve el polinomio interpolador.

    Parámetros:
    - x_values: array-like, puntos x conocidos (usados para ajustar el rango).
    - y_values: array-like, valores y conocidos.
    - n: int, número de nodos de Chebyshev a usar.
        Si es None, se usa len(x_values).

    Retorna:
    - x_cheb: nodos de Chebyshev en el intervalo [a, b].
    - f: polinomio interpolador.
    """
    if len(x_values) != len(y_values):
        raise ValueError("x_values e y_values deben tener la misma longitud.")
    if len(x_values) == 0:
        raise ValueError("Los arreglos no pueden estar vacíos.")

    if n is None:
        n = len(x_values)
    if n <= 0:
        raise ValueError("n debe ser al menos 1.")
    if n > len(x_values):
        raise ValueError("n no puede ser mayor que cantidad de puntos originales.")

    a, b = min(x_values), max(x_values)

    # Genera nodos de Chebyshev en [a, b]
    k = np.arange(1, n + 1)
    cheb_nodes = np.cos((2 * k - 1) * np.pi / (2 * n))  # Nodos en [-1, 1]
    x_cheb = 0.5 * (a + b) + 0.5 * (b - a) * cheb_nodes  # Mapea a [a, b]

    # Evaluar en nodos de Chebyshev usando splines cúbicos (estable)
    _, f_interp = polynomialInterpolation(x_values, y_values)
    y_cheb = f_interp(x_cheb)

    # Ajustar polinomio de Chebyshev
    poly = np.polynomial.Chebyshev.fit(x_cheb, y_cheb, deg=n-1)

    return x_cheb, poly
