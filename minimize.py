import cvxpy as cp
import numpy as np

def optimizar_ventas(prevision_ventas, produccion_real, penalizacion_kwh, graph=False):
    # Parámetros de la planta
    n = len(prevision_ventas)  # Número de horas
    S_max = 3600  # Capacidad máxima de la batería (3.600 kWh)
    B_max = 1800  # Máxima carga/descarga por hora (1.800 kWh)
    V_max = 4000  # Máxima energía que puede venderse a la red por hora (4.000 kWh)
    S_init = 1080  # Estado de la batería al inicio (30% de 3.600 kWh)
    S_final = 1080  # Estado de la batería al final (30% de 3.600 kWh)

    # Variables de optimización
    v_ajustada = cp.Variable(n)  # Ventas ajustadas
    b_in = cp.Variable(n)        # Energía que entra a la batería
    b_out = cp.Variable(n)       # Energía que sale de la batería
    s = cp.Variable(n)           # Estado de carga de la batería
    carga_flag = cp.Variable(n, boolean=True)  # Indicador de carga

    # Restricciones
    constraints = []

    # Estado de la batería al inicio y final del día
    constraints.append(s[0] == S_init)
    constraints.append(s[-1] == S_final)

    for t in range(1, n):
        # Restricciones de energía
        constraints.append(s[t] == s[t-1] + b_in[t] - b_out[t])
        constraints.append(0 <= s[t])  # El estado de la batería debe ser positivo
        constraints.append(s[t] <= S_max)  # No puede exceder la capacidad máxima de la batería
        
        # Restricciones de carga/descarga
        constraints.append(0 <= b_in[t])  # Carga no negativa
        constraints.append(b_in[t] <= B_max)  # No puede superar la capacidad de carga por hora
        constraints.append(0 <= b_out[t])  # Descarga no negativa
        constraints.append(b_out[t] <= B_max)  # No puede superar la capacidad de descarga por hora
        
        # Restricciones de venta
        constraints.append(0 <= v_ajustada[t])  # Las ventas no pueden ser negativas
        constraints.append(v_ajustada[t] <= V_max)  # No puede exceder la capacidad máxima de venta por hora
        
        # Relación entre ventas ajustadas y producción
        constraints.append(v_ajustada[t] == produccion_real[t] + (s[t-1] - s[t]))  # Ventas = Producción + Cambio de estado de batería

    # Penalización por desviación entre ventas previstas y ventas ajustadas
    penalizacion = cp.sum_squares(v_ajustada - prevision_ventas) * penalizacion_kwh

    # Función objetivo: minimizar la penalización
    objective = cp.Minimize(penalizacion)

    # Resolver el problema
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GUROBI)

    if graph:
        import matplotlib.pyplot as plt

        # Graficar resultados
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(range(n), v_ajustada.value, label='Ventas Ajustadas', color='g')
        ax.plot(range(n), prevision_ventas, label='Ventas Previstas', color='r', linestyle='dashed')
        plt.grid()
        plt.legend()
        plt.show()

    return v_ajustada.value, problem.value
