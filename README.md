# Predictor de ventas — AdventureWorks
### Proyecto Grupal 12 — Deep Learning
**Integrantes:** Federico Bidegaray · Maria Noelia Taboada · Luis Viana

---

## Descripción

Predicción de la cantidad de unidades vendidas (`CantidadVendida`) por línea de pedido, sobre
el dataset de ventas de AdventureWorks, usando una red neuronal profunda (DNN) con *entity
embeddings* para representar productos de alta cardinalidad.

Este proyecto es la culminación de tres prácticas individuales del curso (MLP → DNN con
embeddings e historial → transfer learning y explicabilidad), y agrega un **prototipo
interactivo** que permite predecir en tiempo real a partir de un producto, un canal de venta
y si hay una oferta activa.

**Video de presentación:** _[completar con el link al video una vez subido]_

---

## Problema y arquetipo de producto ML

- **Problema:** predecir cuántas unidades se venderán en una línea de pedido, a partir de
  variables del producto, el cliente, la tienda y las condiciones de venta.
- **Por qué Deep Learning:** el problema combina variables categóricas de alta cardinalidad
  (cientos de productos), variables numéricas, y patrones no lineales entre ellas — un caso
  donde una red con *entity embeddings* aprovecha mejor la estructura de los datos que un
  modelo lineal o reglas fijas.
- **Arquetipo de producto ML:** Software 2.0 — el modelo reemplaza estimaciones manuales o
  reglas fijas por un sistema entrenado directamente sobre el historial de ventas.

---

## Arquitectura del modelo

DNN con dos ramas que se concatenan:
- **Embeddings de entidad** para `producto` y `subcategoria` (en vez de one-hot, por su alta
  cardinalidad).
- **Variables tabulares** (numéricas estandarizadas + categóricas de baja cardinalidad en
  one-hot + variables de historial de cliente/tienda calculadas solo con el conjunto de
  entrenamiento, sin fuga de datos).

Bloque denso final: 128 → 64 → 32 → 16 → 1, con `BatchNormalization` y `Dropout`.

**Resultado en prueba:** MAE 0,43 · RMSE 1,02 · R² 0,74 (mejora de ~28% en MAE respecto del
MLP de línea base).

---

## Cómo reproducirlo

1. Abrir `Entrenar_y_Exportar_Modelo.ipynb` en [Google Colab](https://colab.research.google.com/).
2. El notebook busca automáticamente `ventas_adventureworks.csv` en tu Google Drive. Si no lo
   encuentra (por ejemplo, si estás corriendo esto desde una cuenta distinta a la del grupo),
   te va a pedir subirlo manualmente — el archivo está incluido en este mismo repositorio
   como `ventas_adventureworks.csv.gz` (comprimido, por el límite de tamaño de GitHub;
   pandas lo lee directo, sin necesidad de descomprimirlo a mano).
3. Ejecutar todas las celdas en orden, de punta a punta.
4. La última celda instala y lanza una interfaz de **Gradio**, que genera un link público
   temporal (`https://xxxxx.gradio.live`, válido ~72 horas) para probar el predictor de forma
   interactiva.

> **Nota:** el link público de Gradio expira a las ~72 horas de generado y depende de que la
> celda de Colab siga corriendo. Cualquiera (incluido el docente) puede reproducirlo desde
> cero: no depende de accesos ni archivos que solo tenga el grupo — todo lo necesario
> (notebook + dataset) está en este repositorio.

---

## Estructura de este repositorio

```
├── Entrenar_y_Exportar_Modelo.ipynb   # Notebook único: datos, entrenamiento y demo con Gradio
├── ventas_adventureworks.csv.gz       # Dataset comprimido (28 MB → 1,8 MB)
├── README.md                          # Este archivo
```

---

## Resumen del ciclo de vida del proyecto

| Etapa | Contenido |
|---|---|
| Práctica 1 | Línea base: MLP simple |
| Práctica 2 | DNN con *entity embeddings* + variables de historial de cliente/tienda (modelo usado en este prototipo) |
| Práctica 3 | Transfer learning (embeddings de texto pre-entrenados) y explicabilidad (SHAP) |
| Proyecto grupal | Prototipo interactivo del modelo de Práctica 2, con interfaz Gradio |

## Desafíos y limitaciones conocidas

- El modelo subestima sistemáticamente las cantidades de venta altas (poco frecuentes en los
  datos históricos) — documentado y analizado en el informe de Práctica 2.
- El formulario del prototipo pide solo 3 campos (producto, canal, oferta); el resto de las
  variables que el modelo necesita se completan con valores típicos calculados del conjunto
  de entrenamiento (precio mediano por producto, historial promedio de cliente/tienda, fecha
  actual), no con datos reales de un cliente o tienda específicos.
- Para un despliegue real se necesitaría: acceso en vivo a estas variables desde un sistema
  operacional, monitoreo continuo del modelo (detección de *drift*), y una URL persistente
  (este prototipo usa un link temporal de Gradio por simplicidad y costo cero).
