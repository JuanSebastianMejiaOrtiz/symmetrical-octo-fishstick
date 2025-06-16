import funcsAndRestrictions as fr
import numpy as np
from scipy.optimize import minimize

x0 = np.array([200, 5, 100])  # Punto central


# Función para identificar restricciones activas
def identificar_restricciones_activas(x, tol=1e-5):
    activas = []
    nombres = ['Energía', 'Rendimiento', 'Seguridad', 'Balance']
    restricciones = [
        fr.restriccionEnergia(x),
        fr.restriccionRendimiento(x),
        fr.restriccionSeguridad(x),
        fr.restriccionBalance(x)
    ]

    for i, r in enumerate(restricciones):
        if abs(r) < tol:
            activas.append(nombres[i])
    return activas


# Solucionar optimización para cada criterio
def resolver_optimizacion(metodo='SLSQP'):
    resultados = {}

    # Configuración común
    config = {
        'method': metodo,
        'bounds': fr.limites,
        'constraints': fr.restricciones,
        'options': {'maxiter': 1000, 'ftol': 1e-6}
    }

    # a. Maximizar Ganancia
    res_ganancia = minimize(lambda x: -fr.ganancia(x), x0, **config)
    resultados['Ganancia'] = {
        'x': res_ganancia.x,
        'valor': fr.ganancia(res_ganancia.x),
        'restricciones_activas': identificar_restricciones_activas(res_ganancia.x)
    }

    # b. Minimizar Impacto Ambiental
    res_impacto = minimize(fr.impactoAmbiental, x0, **config)
    resultados['Impacto'] = {
        'x': res_impacto.x,
        'valor': fr.impactoAmbiental(res_impacto.x),
        'restricciones_activas': identificar_restricciones_activas(res_impacto.x)
    }

    # c. Maximizar Pureza
    res_pureza = minimize(lambda x: -fr.pureza(x), x0, **config)
    resultados['Pureza'] = {
        'x': res_pureza.x,
        'valor': fr.pureza(res_pureza.x),
        'restricciones_activas': identificar_restricciones_activas(res_pureza.x)
    }

    return resultados


# Ejecutar y mostrar resultados
if __name__ == "__main__":
    resultados = resolver_optimizacion()

    # Imprimir resultados detallados
    for criterio, datos in resultados.items():
        print(f"\n--- RESULTADO PARA {criterio.upper()} ---")
        print(
            f"Variables óptimas: x1 = {datos['x'][0]:.2f}, x2 = {datos['x'][1]:.2f}, x3 = {datos['x'][2]:.2f}")
        print(f"Valor óptimo: {np.fabs(datos['valor']):.4f}")
        print(
            f"Restricciones activas: {', '.join(datos['restricciones_activas']) or 'Ninguna'}")

    # Análisis de trade-offs
    print("\n\n--- ANÁLISIS DE TRADE-OFFS ---")
    g_val = resultados['Ganancia']['valor']
    e_val = resultados['Impacto']['valor']
    p_val = resultados['Pureza']['valor']
    print(f"Ganancia máxima: {-g_val:.2f} millones USD")
    print(f"Impacto mínimo: {e_val:.2f} eco-puntos")
    print(f"Pureza máxima: {p_val:.2f}%")
    print(f"Trade-off Ganancia vs Impacto: {-g_val/e_val:.2f} USD/eco-punto")
    print(f"Trade-off Pureza vs Impacto: {p_val/e_val:.2f} %/eco-punto")


def analizar_sensibilidad(metodo='SLSQP', jac=None):
    # Probar diferentes puntos iniciales
    puntos_iniciales = [
        np.array([150, 2, 50]),   # Frontera inferior
        np.array([250, 8, 150]),  # Frontera superior
        np.array([180, 4, 80]),   # Punto aleatorio 1
        np.array([220, 6, 120])   # Punto aleatorio 2
    ]

    print(f"\n--- ANÁLISIS DE SENSIBILIDAD - METODO {metodo.upper()} ---")
    for i, punto in enumerate(puntos_iniciales):
        res = minimize(lambda x: -fr.ganancia(x), punto,
                       method=metodo, bounds=fr.limites,
                       constraints=fr.restricciones,
                       jac=jac)
        print(f"Punto inicial {i+1}: Ganancia = {-fr.ganancia(res.x):.2f} "
              f"con x={res.x.round(2)}")

    # Probar con restricción de energía modificada
    restricciones_mod = fr.restricciones.copy()
    restricciones_mod[0] = {'type': 'ineq',
                            'fun': lambda x: 0.01*x[0]**2 + 0.02*x[2]**2 - 600}

    res_mod = minimize(lambda x: -fr.ganancia(x), x0,
                       method=metodo, bounds=fr.limites,
                       constraints=restricciones_mod,
                       jac=jac)
    print("\nCon restricción de energía modificada:")
    print(f"Ganancia: {-fr.ganancia(res_mod.x):.2f} con x={res_mod.x.round(2)}")


def jac_ganancia(x):
    x1, x2, x3 = x
    df_dx1 = -0.1*(x1-200) + 0.001*x2*x3 - 0.001*x1*x3
    df_dx2 = -4*(x2-5) + 0.001*x1*x3
    df_dx3 = -0.2*(x3-100) + 0.001*x1*x2 - 0.0005*x1**2
    return np.array([-df_dx1, -df_dx2, -df_dx3])


analizar_sensibilidad(metodo='SLSQP')
analizar_sensibilidad(metodo='SLSQP', jac=jac_ganancia)

analizar_sensibilidad(metodo='trust-constr')
analizar_sensibilidad(metodo='trust-constr', jac=jac_ganancia)
