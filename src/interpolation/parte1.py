import matplotlib.pyplot as plt
import numpy as np
from chebyshev import chebyshev_interpolation
from cubicSplines import cubicSplinesInterpolation
from polynomial import polynomialInterpolation
from loadSensorData import getSensorData


def plot_interpolation(x, f, label, sensor):
    """
    Grafica datos y función de interpolación
    """
    x_eval = np.linspace(min(x), max(x), 500)
    plt.plot(x_eval, f(x_eval), label=label)
    plt.xlabel(r'Temperatura ($^\circ$C)')
    plt.ylabel(r'Voltaje ($mV$)')
    plt.title(f"Interpolación - Sensor {sensor}")
    plt.legend()
    plt.grid(True)


def analyze_sensor(sensor_id):
    """
    Realiza análisis completo para un sensor
    """
    # Cargar datos
    file_path = f"./public/sensorData/sensor{sensor_id}.csv"
    temperature, voltage = getSensorData(file_path)

    # Aplicar diferentes métodos de interpolación
    coeff, f_poly = polynomialInterpolation(temperature, voltage)
    nodes, f_cheb = chebyshev_interpolation(temperature, voltage)
    f_spline = cubicSplinesInterpolation(temperature, voltage)

    print(f"Sensor {sensor_id}:")
    print(f"Coeficientes polinomiales: {coeff}")
    print(f"Nodos de Chebyshev: {nodes}")
    print("\n")

    # Crear figura
    plt.figure(figsize=(10, 6))

    # Graficar cada método
    plt.plot(temperature, voltage, 'o', label='Datos')
    plot_interpolation(temperature, f_poly, "Polinomial", sensor_id)
    plot_interpolation(temperature, f_cheb, "Chebyshev", sensor_id)
    plot_interpolation(temperature, f_spline, "Splines cúbicos", sensor_id)


# Análisis para cada sensor
sensors = ['A', 'B', 'C']
for sensor in sensors:
    analyze_sensor(sensor)

plt.show()
