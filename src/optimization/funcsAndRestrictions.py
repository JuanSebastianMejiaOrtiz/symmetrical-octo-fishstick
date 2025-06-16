import numpy as np
from scipy.optimize import Bounds


def ganancia(x):
    x1, x2, x3 = x
    section = -0.05*((x1 - 200)**2) - 2*((x2 - 5)**2) - 0.1*((x3 - 100)**2)
    section2 = 0.001*x1*x2*x3 - 0.0005*(x1**2)*x3
    return 150 + section + section2


def impactoAmbiental(x):
    x1, x2, x3 = x
    section = 0.002*((x1-150)**2) + 0.5*((x2-4)**2) + 0.005*((x3-80)**2)
    section2 = 0.0001*x1*x3
    return 10 + section + section2


def pureza(x):
    x1, x2, x3 = x
    section = - 0.001*((x1-220)**2) - 0.8*((x2-6)**2) - 0.008*((x3-120)**2)
    section2 = 0.0003*x1*x2
    return 90 + section + section2


# Restricciones en formato g(x) <= 0
def restriccionEnergia(x):
    x1, x2, x3 = x
    return 0.01*x1**2 + 0.02*x3**2 - 700


def restriccionSeguridad(x):
    x1, x2, x3 = x
    return (x1-200)**2 + 10*(x2-5)**2 - 300


def restriccionRendimiento(x):
    x1, x2, x3 = x
    return 200 - (0.5*x1 + 10*x2 + 0.1*x3 - 10*np.sin(x2))


def restriccionBalance(x):
    x1, x2, x3 = x
    return x3 + 2*x2 - 0.5*x1


limites = Bounds([150, 2, 50], [250, 8, 150])

restricciones = [
    {'type': 'ineq', 'fun': restriccionEnergia},
    {'type': 'ineq', 'fun': restriccionRendimiento},
    {'type': 'ineq', 'fun': restriccionSeguridad},
    {'type': 'ineq', 'fun': restriccionBalance}
]
