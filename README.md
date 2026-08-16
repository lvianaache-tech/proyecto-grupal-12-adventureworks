[README (1).md](https://github.com/user-attachments/files/31123742/README.1.md)
# Control de Calidad de Cantidades — AdventureWorks
### Proyecto Grupal 12 — Deep Learning
**Integrantes:** Federico Bidegaray · Maria Noelia Taboada · Luis Viana

---

## Descripción

Herramienta de **control de calidad para líneas de pedido**, construida sobre el modelo de
regresión de la Práctica 2 (DNN con *entity embeddings* + variables de historial de
cliente/tienda). El modelo no reemplaza la cantidad que carga una persona — calcula la
cantidad **esperada** para ese cliente, producto y canal, y **marca las líneas que se alejan
mucho de lo esperado**, para que alguien las revise antes de confirmar el pedido (por
ejemplo, un error de tipeo como cargar 40 en lugar de 4).

Este proyecto es la culminación de tres prácticas individuales del curso (MLP → DNN con
embeddings e historial → transfer learning y explicabilidad), reutilizando el mejor modelo
de esa serie (Práctica 2) dentro de un prototipo interactivo con Gradio.

**Video de presentación:** _[completar con el link al video una vez subido]_

---

## Problema y arquetipo de producto ML

- **Problema:** las líneas de pedido se cargan a mano y, a ese volumen, los errores de
  tipeo son inevitables y difíciles de detectar por revisión manual. Se necesita una forma
  de priorizar automáticamente qué líneas revisar, sin frenar la carga de pedidos.
- **Por qué Deep Learning:** la cantidad "esperada" para una línea depende de una
  combinación no lineal de producto, cliente, canal y estacionalidad — no es algo que una
  regla fija (del tipo "avisar si supera 10 unidades") pueda capturar bien.
- **Arquetipo de producto ML: Human-in-the-Loop.** El modelo **prioriza** qué líneas
  conviene mirar; la persona **confirma, corrige o descarta** la alerta. El modelo nunca
  modifica ni bloquea un pedido por sí solo — dado que tiene un error conocido (MAE 0,43
  unidades) y una limitación documentada (subestima pedidos grandes legítimos), automatizar
  la decisión sin supervisión sería imprudente.

---

## Cómo funciona

1. Se elige un **cliente** (su historial de compra se toma automáticamente del sistema, no
   se tipea — evita que alguien lo "ajuste" a mano para esquivar una alerta).
2. Se elige **producto** y **canal de venta**, y se ingresa la **cantidad cargada** a revisar.
3. El modelo calcula la cantidad esperada para esa combinación y la compara contra la
   cargada:
   - **Verde — dentro de lo esperado:** no requiere revisión.
   - **Ámbar — desvío moderado:** vale una revisión rápida.
   - **Rojo — desvío grande:** el patrón típico de un error de carga, revisar con prioridad.
4. Un gráfico adicional muestra **por qué** el modelo espera esa cantidad (efecto del
   historial del cliente y del canal de venta sobre la predicción).

El umbral de alerta es **generoso a propósito**: como el modelo subestima los pedidos
grandes legítimos (una limitación ya documentada en las prácticas anteriores), solo marca
desvíos muy grandes — evita saturar de falsas alarmas a quien revisa.

---

## Arquitectura del modelo

DNN con dos ramas que se concatenan:
- **Embeddings de entidad** para `producto` (266 categorías) y `subcategoria` (35
  categorías), en vez de one-hot, por su alta cardinalidad.
- **Variables tabulares**: numéricas estandarizadas + categóricas de baja cardinalidad en
  one-hot + variables de historial de cliente/tienda (calculadas solo con el conjunto de
  entrenamiento, sin fuga de datos).

Bloque denso final: 128 → 64 → 32 → 16 → 1, con `BatchNormalization` y `Dropout`.

**Resultado de referencia (modelo oficial de Práctica 2, mismo pipeline):**
MAE 0,43 · RMSE 1,02 · R² 0,74 en el conjunto de prueba — una mejora de ~28% en MAE
respecto del MLP de línea base de la Práctica 1.

> **Nota:** el notebook de este prototipo (`Copia_de_Grupo_12_Prototipo_DNN_QC.ipynb`)
> entrena el modelo dentro de la propia app, pero **todavía no incluye una celda de
> evaluación propia** (separación en validación/prueba con MAE/RMSE/R² impresos). El número
> de referencia de arriba corresponde al modelo oficial de Práctica 2 con el mismo pipeline
> de datos y arquitectura — se recomienda agregar esa evaluación al notebook antes de citar
> un MAE específico de esta versión en el informe o el video.

---

## Cómo reproducirlo

1. Abrir `Copia_de_Grupo_12_Prototipo_DNN_QC.ipynb` en [Google Colab](https://colab.research.google.com/).
2. Ejecutar la celda de instalación, y luego la celda de carga de datos: va a pedir subir el
   archivo `ventas_adventureworks.csv.gz` (incluido en este repositorio). **Importante:**
   subir el archivo comprimido tal cual, aunque el destino se llame `.csv` — el código espera
   contenido comprimido (`compression='gzip'`) al leerlo.
3. Ejecutar la celda principal (entrena el modelo, ~1-2 minutos) y la celda final
   (`demo.launch(share=True)`), que genera un link público temporal
   (`https://xxxxx.gradio.live`, válido hasta ~1 semana) para probar la herramienta.
4. Dejar la celda de Colab corriendo mientras se use el link — si se interrumpe o se cierra
   la sesión, el link se cae y hay que volver a correr la celda para obtener uno nuevo.

---

## Estructura de este repositorio

```
├── Copia_de_Grupo_12_Prototipo_DNN_QC.ipynb   # Prototipo final: control de calidad (Human-in-the-Loop)
├── Entrenar_y_Exportar_Modelo.ipynb            # Prototipo alternativo: predictor simple (no usado en el video)
├── Prototipo_simulador_descuentos.ipynb        # Prototipo alternativo: simulador what-if (no usado en el video)
├── ventas_adventureworks.csv.gz                # Dataset comprimido (28 MB → 1,8 MB)
├── README.md                                   # Este archivo
```

---

## Resumen del ciclo de vida del proyecto

| Etapa | Contenido |
|---|---|
| Práctica 1 | Línea base: MLP simple |
| Práctica 2 | DNN con *entity embeddings* + variables de historial de cliente/tienda (modelo usado en este prototipo) |
| Práctica 3 | Transfer learning (embeddings de texto pre-entrenados) y explicabilidad (SHAP) |
| Proyecto grupal | Prototipo de control de calidad Human-in-the-Loop, con el modelo de Práctica 2 |

## Desafíos y limitaciones conocidas

- El modelo subestima sistemáticamente las cantidades de venta altas (poco frecuentes en
  los datos históricos) — documentado en el informe de Práctica 2, y es la razón por la que
  el umbral de alerta de este prototipo es deliberadamente generoso.
- El notebook de este prototipo todavía no incluye una evaluación propia en un conjunto de
  prueba separado (ver nota en "Arquitectura del modelo") — pendiente de agregar.
- Se iteró el caso de uso del prototipo varias veces antes de llegar a este (un predictor
  simple, un simulador de descuentos, y finalmente control de calidad) — ambas versiones
  anteriores quedan en el repositorio como referencia, pero no son las que se muestran en
  el video final.
- Para un despliegue real haría falta: integración en vivo con el sistema de carga de
  pedidos (no un CSV subido a mano), monitoreo continuo del modelo (detectar *drift*), y
  registrar qué alertas confirma o descarta cada persona, para medir si el umbral de aviso
  sigue siendo el correcto con el tiempo.
- Explorado pero no incorporado: un modelo en dos etapas (clasificador "1 vs. más de 1" +
  regresor solo para pedidos mayores), probado en Práctica 3, que mostró resultados
  prometedores para reducir la subestimación de pedidos grandes — queda como línea de
  mejora concreta para una futura iteración.
