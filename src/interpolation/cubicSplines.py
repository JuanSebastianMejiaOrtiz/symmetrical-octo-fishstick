from scipy.interpolate import interp1d


def cubicSplinesInterpolation(x_values, y_values):
    '''
    Realiza interpolación usando splines cubicos y devuelve el polinomio interpolador

    Parámetros:
    - x_values: array-like, puntos x conocidos.
    - y_values: array-like, valores y conocidos.

    Retorna:
    - f: polinomio interpolador
    '''
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length")
    if len(x_values) == 0:
        raise ValueError("The arrays cannot be empty")
    return interp1d(x_values, y_values, kind='cubic')
