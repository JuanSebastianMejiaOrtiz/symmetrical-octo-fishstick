import numpy as np


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
    - coeff: Coeficientes del polinomio interpolador.
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

    # Paso 1: Mapear al intervalo [-1, 1]
    def scale(x):
        return (2 * x - (a + b)) / (b - a)

    x_scaled = scale(np.array(x_values))

    # Paso 2: Calcular bases de Chebyshev (hasta grado n-1)
    def T(k, t):
        return np.cos(k * np.arccos(t))

    # Construir matriz de diseño
    A = np.zeros((len(x_scaled), n))
    for i, x in enumerate(x_scaled):
        for j in range(n):
            A[i, j] = T(j, x)

    # Paso 3: Resolver el sistema lineal
    coeff = np.linalg.solve(A, y_values)

    # Construir función interpolante
    def f(t):
        t_scaled = scale(t)
        result = 0
        for k, c in enumerate(coeff):
            result += c * T(k, t_scaled)
        return result

    return coeff, f
