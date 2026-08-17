[README.md](https://github.com/user-attachments/files/31141138/README.md)
# Proyecto Grupal 12 — Control de calidad de cantidades (AdventureWorks)

Proyecto final del curso de **Deep Learning** (Universidad de Montevideo). Recorre el ciclo
de vida completo de un proyecto de ML/DL sobre datos de ventas de AdventureWorks: desde la
definición del problema y una línea base, hasta un modelo profundo y un **prototipo
funcional** en Gradio.

**Integrantes:** Federico Bidegaray · Maria Noelia Taboada · Luis Viana

**Video de presentación:** _[completar con el link al video una vez subido]_

---

## Problema y caso de uso

El objetivo técnico es estimar la **cantidad vendida** por línea de pedido (regresión) a
partir del producto, el cliente, el canal, la fecha y variables de historial.

El caso de uso del producto **no** es predecir la cantidad (en un pedido real ya viene
dada), sino usar el modelo como **segunda opinión / control de calidad**: para cada línea,
el modelo calcula la cantidad *esperada* y marca aquellas cuya cantidad cargada se aleja
mucho de lo esperado, para que una persona las revise (típicamente, errores de carga como
40 en vez de 4).

**Arquetipo de producto: Human-in-the-Loop.** El modelo prioriza qué revisar; la decisión
final es de quien carga o supervisa el pedido — el modelo nunca modifica ni bloquea nada
por sí solo.

---

## Modelo y resultados

Se probaron modelos de complejidad creciente, todos con la **misma partición cronológica**
(70/15/15) para que la comparación sea justa. Métricas confirmadas sobre el conjunto de
prueba:

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| MLP con one-hot (Práctica 1) | 0,5925 | 1,1146 | 0,6956 |
| **DNN con embeddings + historial (Práctica 2 — modelo del prototipo)** | **0,4282** | **1,0223** | **0,7440** |

En la Práctica 3 se exploraron varias alternativas de transfer learning (embeddings de
texto pre-entrenados con GloVe, en modo congelado y con fine-tuning) y técnicas adicionales
(pérdida Poisson, gradient boosting con LightGBM, un modelo en dos etapas). Ninguna superó
de forma clara y verificada al modelo de la Práctica 2 — los números obtenidos variaron
entre corridas y documentos sin una versión única y confirmada, por lo que **no se reportan
acá como definitivos**. El modelo que efectivamente alimenta el prototipo es el de
**Práctica 2**.

La explicabilidad (SHAP sobre las variantes de Práctica 3, y el gráfico de factores del
propio prototipo) coincide en que el **historial de compra del cliente** es, de lejos, el
factor que más pesa en cada predicción.

---

## Cómo funciona el prototipo

1. Se elige un **cliente** (su historial de compra se toma automáticamente del sistema, no
   se tipea).
2. Se elige **producto** y **canal de venta**, y se ingresa la **cantidad cargada** a revisar.
3. El modelo calcula la cantidad esperada y compara: **verde** (dentro de lo esperado),
   **ámbar** (desvío moderado, revisar), o **rojo** (desvío grande, revisar con prioridad).
4. Un gráfico adicional muestra el efecto del historial del cliente y del canal de venta
   sobre la predicción.

El umbral de alerta es **generoso a propósito**: como el modelo subestima los pedidos
grandes legítimos (ver limitaciones abajo), solo marca desvíos muy grandes, para no saturar
de falsas alarmas a quien revisa.

---

## Cómo ejecutar el prototipo

El prototipo corre en **Google Colab**:

1. Abrir `Grupo_12_Bidegaray_Viana_Taboada.ipynb` en Colab.
2. Ejecutar la celda de instalación (`!pip install gradio tensorflow scikit-learn`).
3. Ejecutar la celda de carga de datos: busca `ventas_adventureworks.csv(.gz)`
   automáticamente en Google Drive; si no lo encuentra, pide subirlo manualmente (está
   incluido en este repositorio como `ventas_adventureworks.csv.gz`).
4. Ejecutar la celda principal: entrena el modelo (~1-2 min) y genera un **link público**
   temporal (`https://xxxxx.gradio.live`, válido hasta ~1 semana) para probar la
   herramienta.

> Dejar la celda de Colab corriendo mientras se usa el link — si se interrumpe o se cierra
> la sesión, el link se cae y hay que volver a correr la celda para obtener uno nuevo.

---

## Estructura de este repositorio

```
├── Grupo_12_Bidegaray_Viana_Taboada.ipynb   # Prototipo final: control de calidad (Human-in-the-Loop)
├── ventas_adventureworks.csv.gz              # Dataset comprimido (28 MB → 1,8 MB)
├── requirements.txt                          # Dependencias de referencia
├── README.md                                 # Este archivo
```

---

## Decisiones metodológicas y limitaciones (honestas)

- **Partición cronológica, no aleatoria:** se entrena con el pasado y se evalúa con el
  futuro, para no filtrar información desde fechas posteriores hacia el entrenamiento.
- **Exclusión de `LineTotal`** como variable predictora: se calcula a partir de la cantidad
  vendida, así que usarlo sería una fuga de datos (la respuesta filtrada en una variable de
  entrada).
- **Variables de historial de cliente/tienda** calculadas solo con el conjunto de
  entrenamiento, para que no se filtre información de validación/prueba hacia atrás.
- **Límite de información, no de modelo:** el modelo subestima sistemáticamente los pedidos
  de volumen medio-alto (4-10 unidades). Se probaron variantes (pérdida distinta,
  arquitecturas más grandes, otras familias de modelo) sin superar ese techo de forma
  consistente — sugiere que faltan variables que anticipen esos pedidos grandes, más que
  una limitación de capacidad del modelo en sí. Por eso el prototipo usa un umbral de
  alerta tolerante en vez de uno ajustado al error promedio.
- **Los descuentos aportan poca señal** en estos datos (una minoría de las ventas tuvo
  descuento activo), por lo que no se usan como palanca principal en el prototipo.
- Antes de llegar a esta versión, se exploraron otros enfoques de producto para el mismo
  modelo (por ejemplo, un simulador de impacto de descuentos) — no se incluyen en este
  repositorio por no ser los que se muestran en el video final.
- **Para producción** haría falta: integración en vivo con el sistema de carga de pedidos
  (no un CSV subido a mano), monitoreo continuo del modelo (detectar *drift* si cambian los
  patrones de venta con el tiempo), y registrar qué alertas confirma o descarta cada
  persona, para medir si el umbral sigue siendo el correcto.

---

## Herramientas

Python · TensorFlow/Keras · scikit-learn · Gradio · pandas · NumPy · Google Colab.

---

## Resumen del ciclo de vida del proyecto

| Etapa | Contenido |
|---|---|
| Práctica 1 | Línea base: MLP simple |
| Práctica 2 | DNN con *entity embeddings* + variables de historial de cliente/tienda (modelo usado en este prototipo) |
| Práctica 3 | Exploración de transfer learning (embeddings de texto pre-entrenados) y explicabilidad (SHAP) |
| Proyecto grupal | Prototipo de control de calidad Human-in-the-Loop, con el modelo de Práctica 2 |
