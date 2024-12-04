import numpy as np

# Tabla de verdad para una puerta lógica (AND en este caso)
X = np.array([
    [1, 1, 1],   # X0, X1, X2
    [1, 1, -1],
    [1, -1, 1],
    [1, -1, -1]
])
y_d = np.array([1, -1, -1, -1])  # Salida esperada (deseada)

# Inicialización de pesos y parámetros
w = np.array([0.6473185, 0.3781776, 0.3316055])  # Pesos iniciales
alpha = 0.4  # Tasa de aprendizaje

# Función de activación: Signo
def activation_function(value):
    return 1 if value >= 0 else -1

# Entrenamiento del perceptrón
iterations = 10  # Número máximo de iteraciones
for epoch in range(iterations):
    print(f"\nIteración {epoch + 1}:")
    error_total = 0
    for i in range(len(X)):
        # Cálculo de la salida del perceptrón
        y = np.dot(w, X[i])  # Producto punto (sumatoria ponderada)
        y_pred = activation_function(y)  # Salida activada

        # Cálculo del error
        error = y_d[i] - y_pred
        error_total += abs(error)
        print(f"  Entrada: {X[i]}, Salida esperada: {y_d[i]}, Predicha: {y_pred}, Error: {error}")

        # Actualización de pesos si hay error
        if error != 0:
            w += alpha * error * X[i]
            print(f"    Pesos actualizados: {w}")

    # Detener si el error total es cero (solución encontrada)
    if error_total == 0:
        print("\nEl perceptrón ha aprendido correctamente los pesos.")
        break
else:
    print("\nNo se alcanzó un error total de cero en las iteraciones permitidas.")

# Pesos finales
print(f"\nPesos finales: {w}")
