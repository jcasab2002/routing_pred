# Modelado de Tiempos de Parada Logísticos

## Descripción del Proyecto

Este proyecto se enfoca en el análisis y modelado de los tiempos de parada de vehículos en operaciones logísticas. El objetivo principal es construir un modelo predictivo robusto que pueda estimar con precisión la duración de una parada, lo que es crucial para la optimización de rutas, la planificación de la capacidad y la mejora de los tiempos de entrega.

El enfoque adoptado es un **Modelo de Dos Partes (Two-Part Model)**, diseñado para manejar la distribución asimétrica y bimodal de los tiempos de parada:
1.  **Modelo de Clasificación:** Una regresión logística que distingue entre paradas muy cortas (potencialmente atípicas) y paradas de duración normal.
2.  **Modelo de Regresión:** Un Modelo Lineal Generalizado (GLM) con distribución Gamma, entrenado solo en las paradas "normales", para predecir su duración real.

---

##  Estructura del Proyecto

La estructura del proyecto está organizada en los siguientes archivos y carpetas:

* `main_definitivo.py`: Script principal para ejecutar el pipeline completo de carga de datos, pre-procesamiento, entrenamiento de modelos y validación.
* `data_load.py`: Contiene funciones para cargar los archivos de datos (en formato `.xlsx`).
* `data_processor.py`: Lógica para limpiar, enriquecer y transformar los datos brutos en el formato adecuado para el modelado.
* `models.py`: Implementación de los modelos predictivos (`TwoPartModel`, `GLMGamlogModel`, etc.).
* `modelsv2.py` : Modelos deprecados
* `vis.py`: Contiene la clase `DataFrameAnalyzer` y funciones para la visualización de datos y los diagnósticos del modelo (ej. histogramas, curva ROC).
* `pruebas/pruebas`: **Carpeta dedicada para los archivos de entrada.** Aquí se deben almacenar los archivos `.xlsx` de `perfiles`, `paradas` y `eventos` tanto para el entrenamiento como para la validación. Esta carpeta fue proporcionada por la contraparte
* `README.md`: Este archivo.
* `requirements.txt`: Obtenido con pip freeze. 

## Que correr?
* `main_definitivo.py`: Implementacion del two stage modelling.
*  `main.py`: Implementacion del GLM de primera instancia, sin analisis ni separacion de outliers (fue la primera version).

---
