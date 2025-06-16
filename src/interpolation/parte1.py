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


def evaluate_interpolation(f, x_eval):
    """
    Evalua la función de interpolación en un punto x
    """
    return f(x_eval)


def analyze_sensor(sensor_id):
    """
    Realiza análisis completo para un sensor
    """
    # Cargar datos
    file_path = f"./public/sensorData/sensor{sensor_id}.csv"
    temperature, voltage = getSensorData(file_path)

    # Aplicar diferentes métodos de interpolación
    coeff, f_poly = polynomialInterpolation(temperature, voltage)
    coeff, f_cheb = chebyshev_interpolation(temperature, voltage)
    f_spline = cubicSplinesInterpolation(temperature, voltage)

    print(f"Sensor {sensor_id}:")
    print(f"Coeficientes polinomiales: {coeff}")
    print(f"Coeficientes Chebyshev: {coeff}")
    print("\n")

    # Crear figura
    plt.figure(figsize=(10, 6))

    # Graficar cada método
    plt.plot(temperature, voltage, 'o', label='Datos')
    plot_interpolation(temperature, f_poly, "Polinomial", sensor_id)
    plot_interpolation(temperature, f_cheb, "Chebyshev", sensor_id)
    plot_interpolation(temperature, f_spline, "Splines cúbicos", sensor_id)

    functions = {"polinomial": f_poly, "chebyshev": f_cheb, "cubicSplines": f_spline}
    return functions


# Análisis para cada sensor
sensors = ['A', 'B', 'C']
functions = {}
for sensor in sensors:
    functions[sensor] = analyze_sensor(sensor)

# Evaluación de funciones
temperatures = [25, 55, 95]
for temperature in temperatures:
    print(f"Temperatura evaluada: {temperature} °C")
    for sensor in sensors:
        print(f"Sensor {sensor}:")
        print(f"Polinomial: {evaluate_interpolation(functions[sensor]['polinomial'], temperature):.2f}")
        print(f"Chebyshev: {evaluate_interpolation(functions[sensor]['chebyshev'], temperature):.2f}")
        print(f"Splines cúbicos: {evaluate_interpolation(functions[sensor]['cubicSplines'], temperature):.2f}")
        print("\n")

plt.show()
