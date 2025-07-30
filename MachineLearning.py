# Importar librerías esenciales
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score 

# Cargar dataset Iris (incluido en Scikit-Learn)
iris = datasets.load_iris()
X = iris.data  # Características: largo/ancho de sépalos y pétalos
y = iris.target  # Variable objetivo: especie (0, 1, 2)

# Explorar estructura
print("Dimensiones de X:", X.shape)  # (150 muestras, 4 características)
print("Clases:", np.unique(y))  # [0, 1, 2] → 3 especies

# 80% entrenamiento, 20% prueba (semilla aleatoria=42 para reproducibilidad)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Muestras entrenamiento: {X_train.shape[0]}, prueba: {X_test.shape[0]}")

# Inicializar modelo KNN (vecinos más cercanos)
modelo_knn = KNeighborsClassifier(n_neighbors=3)  # Usar 3 vecinos

# Entrenar con datos de entrenamiento
modelo_knn.fit(X_train, y_train)
print("✅ Modelo entrenado!")

# Predecir en datos de prueba
y_pred = modelo_knn.predict(X_test)

# Calcular precisión
precision = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {precision:.2%}")  # Ej: 96.67%

# Mostrar predicciones vs reales
print("\nEjemplo de predicciones:")
print("Real:    ", y_test[:5])    # Valores reales
print("Predicho:", y_pred[:5])    # Predicciones