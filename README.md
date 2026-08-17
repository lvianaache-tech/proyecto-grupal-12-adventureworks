[README.md](https://github.com/user-attachments/files/31140451/README.md)
# Proyecto Grupal 12 — Estimación de cantidad y control de calidad de pedidos (AdventureWorks)

Proyecto final del curso de **Deep Learning** (Universidad de Montevideo). Recorre el ciclo de vida completo de un proyecto de ML/DL sobre datos de ventas de AdventureWorks: desde la definición del problema y una línea base, hasta un modelo profundo, explicabilidad y un **prototipo funcional** en Gradio.

**Integrantes:** Luis Viana · Federico Bidegaray · Maria Noelia Taboada

---

## Problema y caso de uso

El objetivo técnico es estimar la **cantidad vendida** por línea de pedido (regresión) a partir del producto, el cliente, el canal, la fecha y variables de historial.

El caso de uso del producto **no** es predecir la cantidad (en un pedido real ya viene dada), sino usar el modelo como **segunda opinión / control de calidad**: para cada línea, el modelo calcula la cantidad *esperada* y marca aquellas cuya cantidad cargada se aleja mucho de lo esperado, para que una persona las revise (típicamente, errores de carga como 40 en vez de 4).

**Arquetipo de producto: Human-in-the-Loop.** El modelo prioriza qué revisar; la decisión final es del planificador.

---

## Modelo y resultados

Se probaron modelos de complejidad creciente, todos con la **misma división cronológica** (70/15/15) para una comparación justa. Métricas sobre el conjunto de prueba:

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Línea base (mediana) | 0,76 | 2,16 | — |
| MLP con one-hot (Práctica 1) | 0,59 | 1,11 | 0,70 |
| **DNN con embeddings + historial (Práctica 2)** | **0,43** | **1,02** | **0,74** |
| Transfer learning GloVe + SHAP (Práctica 3) | ~0,43 | ~1,01 | ~0,75 |

El modelo final (**DNN con embeddings de entidad + variables de historial de cliente/tienda**) es el que alimenta el prototipo. La explicabilidad con **SHAP** mostró que el factor que más pesa en cada predicción es el historial de compra del cliente.

---

## Estructura del repositorio

```
proyecto-grupal-12-adventureworks/
├── README.md
├── requirements.txt
├── data/
│   └── ventas_adventureworks.csv        # dataset (ya subido)
├── notebooks/
│   ├── practica1_mlp.ipynb              # línea base + MLP
│   ├── practica2_dnn_embeddings.ipynb   # DNN embeddings + historial (mejor modelo)
│   └── practica3_transfer_shap.ipynb    # transfer learning (GloVe) + explicabilidad SHAP
├── informes/
│   ├── Informe_Practica1.pdf
│   ├── Informe_Practica2.pdf
│   └── Informe_Practica3.pdf
├── prototipo/
│   ├── app_dnn_p2.py                    # app Gradio (control de calidad, modelo P2)
│   └── prototipo_control_calidad.ipynb  # versión Colab, lista para ejecutar
└── video/
    └── enlace_video.md                  # link a la presentación
```

---

## Cómo ejecutar el prototipo

El prototipo corre en **Google Colab** (recomendado, por memoria):

1. Abrir `prototipo/prototipo_control_calidad.ipynb` en Colab.
2. Ejecutar la celda de instalación: `!pip install gradio tensorflow scikit-learn`.
3. Subir `ventas_adventureworks.csv` cuando lo pida.
4. Ejecutar la celda de la app: entrena el modelo (~1-2 min) y genera un **link público** (`share=True`).

En la interfaz se elige un cliente (su historial se toma automáticamente), un producto y un canal, se ingresa la cantidad cargada, y la herramienta muestra un semáforo **coincide / revisar**, la cantidad esperada y el porqué.

---

## Decisiones metodológicas y limitaciones (honestas)

- **División cronológica**, no aleatoria: se entrena con el pasado y se evalúa con el futuro, para no filtrar información.
- **Exclusión de `LineTotal`**: se quitó por ser una fuga de datos (contiene la respuesta).
- **Variables de historial** calculadas solo con el conjunto de entrenamiento, para no filtrar hacia validación/prueba.
- **Límite de señal, no de modelo:** el modelo subestima los pedidos de volumen medio-alto (4–10 unidades). Se probó corregirlo cambiando la pérdida, agrandando la red, con gradient boosting y con un modelo en dos etapas; ninguno rompió ese techo. Faltan variables que anticipen esos pedidos, no capacidad de modelo. Por eso el producto usa un **umbral tolerante**: solo marca desvíos muy grandes.
- **Los descuentos casi no tienen señal** en estos datos (solo el 4,5 % de las ventas tuvo descuento), por lo que no se usan como palanca del prototipo.
- **Deriva temporal (drift):** ya observada entre validación (R² 0,79) y prueba (R² 0,74); en producción requeriría monitoreo continuo.

---

## Herramientas

Python · TensorFlow/Keras · scikit-learn · LightGBM · SHAP · embeddings GloVe (gensim) · Gradio · pandas · matplotlib.

---

## Video de presentación

Enlace: _(agregar en `video/enlace_video.md` o aquí)_
