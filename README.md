

# 🧠 Kohonen 3D - AfroMNIST

Este proyecto implementa una **red de mapas autoorganizados (Self-Organizing Map - SOM)** en 3D utilizando C++, entrenada sobre el dataset **AfroMNIST**, una variante del clásico MNIST. La red agrupa y visualiza imágenes de dígitos escritos con estilo africano en un espacio tridimensional.

## 📁 Estructura de Archivos

| Archivo             | Propósito general                              |
| ------------------- | ---------------------------------------------- |
| `kohonen.h`         | Declaración de la clase `Kohonen` (SOM 3D).    |
| `kohonen.cpp`       | Implementación de la lógica de entrenamiento.  |
| `main.cpp`          | Ejecución del entrenamiento y control general. |
| `visualizacion.cpp` | Visualización y exportación de resultados.     |

---

## 🔧 kohonen.h — Definición del modelo

Este archivo define la clase `Kohonen`, que encapsula:

* 🧩 **Parámetros** de configuración: tamaño de la cuadrícula 3D (`gridX`, `gridY`, `gridZ`), dimensión de entrada (`inputSize`), tasa de aprendizaje, número de épocas, etc.
* 🧠 **Estructura de datos**:

  * `Image`: representa cada imagen como un vector de píxeles normalizados.
  * `weights_`: pesos de las neuronas organizados en un arreglo 3D.
* 🔍 **Métodos clave**:

  * `loadData()`: carga las imágenes del dataset.
  * `train()`: entrena la red usando el algoritmo SOM clásico.
  * `trainWithBatches()`: variante con entrenamiento por lotes.
  * `findBestMatchingUnit()`: busca la neurona más similar.
  * `updateWeights()`: actualiza los pesos usando la vecindad de Kohonen.
  * `saveWeightsForVisualization()`: guarda resultados para visualizar con herramientas externas.

---

## ⚙️ kohonen.cpp — Lógica de entrenamiento

Implementa el algoritmo completo de entrenamiento SOM:

1. **Inicialización aleatoria** de los pesos con `initializeWeights()`.
2. **Distancia Euclidiana** para encontrar la neurona ganadora o *Best Matching Unit* (BMU).
3. **Función de vecindad** gaussiana (`neighborhoodFunction`) que disminuye con la distancia.
4. **Actualización de pesos** proporcional a la distancia al BMU y a la tasa de aprendizaje.
5. **Decaimiento** de la tasa de aprendizaje y sigma con el tiempo.

---

## 🚀 main.cpp — Ejecución del sistema

Este archivo contiene la función `main()` y se encarga de:

* Inicializar la red `Kohonen` con parámetros dados.
* Cargar los datos de AfroMNIST.
* Iniciar el entrenamiento.
* Llamar a la función de guardado de pesos.
* Imprimir resultados o métricas simples por consola.

Opcionalmente puedes modificar los valores de `epochs`, `batchSize`, `learningRate`, etc., directamente aquí.

---

## 📊 visualizacion.cpp — Visualización

Este archivo permite:

* Tomar el archivo de pesos exportado por `saveWeightsForVisualization()`.
* Mostrar o procesar la organización espacial de los dígitos.
* Exportar datos en formatos legibles por herramientas como Python, Excel, o entornos 3D.

Opcionalmente puedes usar bibliotecas como **Matplotlib (Python)** o **Three.js** (web) para visualizar la estructura tridimensional.

---

## ▶️ Cómo compilar y ejecutar

```bash
g++ -std=c++17 -O3 -march=native -funroll-loops main.cpp kohonen.cpp visualizacion.cpp -o kohonen
./kohonen
```

Requiere un archivo `.csv` o `.txt` con los datos normalizados de AfroMNIST.

---

## 📦 Requisitos

* C++17
* Dataset AfroMNIST en formato vectorial
* Compilador compatible con STL (`g++`, `clang++`)

---
