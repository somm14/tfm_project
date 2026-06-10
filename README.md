# FinWellsor - Predicción del riesgo de estrés financiero en asalariados de la Comunidad de Madrid

> **TFM · Máster de Data Science & IA**  
> Proyecto end-to-end de Machine Learning sobre microdatos ECV 2025 (INE)

---

## Índice

1. [Descripción del proyecto](#1-descripción-del-proyecto)
2. [Arquitectura del proyecto](#2-arquitectura-del-proyecto)
3. [Estructura de carpetas](#3-estructura-de-carpetas)
4. [Datos](#4-datos)
5. [Pipeline end-to-end](#5-pipeline-end-to-end)
6. [Modelado](#6-modelado)
7. [Aplicación Streamlit (FinWellsor)](#7-aplicación-streamlit-finwellsor)
8. [Instalación y ejecución](#8-instalación-y-ejecución)
9. [Dependencias](#9-dependencias)
10. [Resultados](#10-resultados)

---
## Motivación

El **estrés financiero** es uno de los **principales factores de riesgo** para la **salud mental** y el **rendimiento laboral**. Sin embargo, su detección en el entorno corporativo sigue siendo reactiva: los departamentos de RRHH actúan cuando el trabajador ya ha manifestado dificultades, no antes.

Disponer de un modelo predictivo permite identificar perfiles de riesgo con antelación, priorizando intervenciones (planes de ahorro, asesoramiento financiero, beneficios flexibles) en la población más vulnerable. En el contexto de la Comunidad de Madrid -economía de alta renta media pero con fuerte presión inmobiliaria y desigualdad creciente- esta capacidad predictiva resulta especialmente relevante.

---

## 1. Descripción del proyecto

Este proyecto predice el **riesgo de estrés financiero alto** en asalariados en activo de la Comunidad de Madrid, utilizando los microdatos de la **Encuesta de Condiciones de Vida 2025 (ECV) del INE**.

El estrés financiero alto se define como la presencia simultánea de **≥ 2 de 5 condiciones**:

| Condición | Variable ECV |
|-----------|-------------|
| Retrasos en hipoteca / alquiler | `HS011` |
| Retrasos en el pago de facturas | `HS021` |
| Retrasos en deudas no vivienda | `HS031` |
| Incapacidad de afrontar gastos imprevistos | `HS060` |
| Dificultad para llegar a fin de mes | `HS120` |

El proyecto es **end-to-end**: parte de los CSVs brutos del INE, construye el dataset analítico, entrena el modelo y lo sirve a través de una aplicación Streamlit con dos roles de usuario (Empleado y RRHH).

---

## 2. Arquitectura del proyecto

El proyecto sigue la **arquitectura medallón** (Bronze → Silver → Gold) para la gestión de datos:

```
Bronze  →  Silver  →  Gold  →  Modelo  →  App Streamlit
  │            │          │        │
Raw CSVs   Limpio +    Features   LightGBM     FinWellsor
 INE       spliteado  preprocesadas  .pkl     (Empleado / RRHH)
           train/test  (train/test)
```

---

## 3. Estructura de carpetas

```
tfm_project/
│
├── src/
│   ├── data/
│   │   ├── 01_bronze/                      # Datos originales descargados del INE
│   │   │   ├── 01_datos_hogar.csv          # Fichero D: datos básicos hogar
│   │   │   ├── 02_detalles_hogar.csv       # Fichero H: renta, vivienda, bienestar
│   │   │   ├── 03_detalles_adulto.csv      # Fichero P: trabajo, educación, salud
│   │   │   ├── 04_datos_persona.csv        # Fichero R: demográficos + vars derivadas
│   │   │   └── info/                       # Documentación auxiliar del INE
│   │   │
│   │   ├── 02_silver/                      # Datos limpios, decodificados y spliteados
│   │   │   ├── dataset_analitico.csv       # Dataset completo antes del split
│   │   │   ├── train_silver/
│   │       │   ├── train_silver.csv        # 80% estratificado (2.357 filas × 67 cols)
│   │   │   │   └── salarios_ocupacion.csv  # Dataset filtrado usado en streamlit
│   │   │   └── test_silver/
│   │   │       └── test_silver.csv         # 20% estratificado (holdout)
│   │   │
│   │   └── 03_gold/                        # Features seleccionadas en el EDA y preprocesadas
│   │       ├── raw/
│   │       │   ├── FEATURES_NUM.txt        # Lista de features numéricas
│   │       │   └── FEATURES_CAT.txt        # Lista de features categóricas
│   │       ├── train_gold/                 # Features preprocesadas - train
│   │       └── test_gold/                  # Features preprocesadas - test
│   │
│   ├── models/
│   │   ├── lg_optimo.pkl                   # Modelo final (Pipeline sklearn + LightGBM)
│   │   └── metadata.json                   # Metadatos del modelo y métricas
│   │
│   ├── img/                                # Imágenes generadas en los notebooks
│   ├── notebooks/
│   │   ├── 01_eda.ipynb                    # Análisis Exploratorio de Datos
│   │   └── 02_modelado.ipynb               # Pipeline de modelado y evaluación
│   │
│   ├── scripts/
│   │   ├── bronze_to_silver.py             # Pipeline de limpieza Bronze → Silver
│   │   └── model_pipeline.py               # Entrenamiento y exportación del modelo
│   │
│   └── utils/
│       ├── constants_utils.py          # Rutas, paleta de colores, constantes
│       ├── functions_utils.py          # Funciones auxiliares (carga CSV, EDA)
│       ├── mapeo_utils.py              # Mapeos de variables ECV, decodificaciones
│       └── visualizations_utils.py     # Funciones de visualización para el EDA
│
├── app.py                              # Aplicación Streamlit (FinWellsor)
├── main.py                             # Punto de entrada del pipeline completo
└── pyproject.toml                      # Configuración del entorno uv
```

---

---

## 4. Datos

**Fuente:** Encuesta de Condiciones de Vida 2025 - INE  
**Población objetivo:** Asalariados en activo residentes en la Comunidad de Madrid  
**Filtros aplicados:**
- `DB040 == 'ES30'` → Comunidad de Madrid
- `PL032 == 1` → Trabajando actualmente
- `PL040A == 3` → Asalariado (empleo actual)

**Tamaño final del dataset:** 2.947 registros × 67 variables  
**Desbalanceo del target:** 84% sin estrés (clase 0) / 16% estrés alto (clase 1) - ratio 1:5

### Grupos de variables

| Grupo | Variables clave |
|-------|----------------|
| Demográficas | edad, sexo, país de nacimiento, nacionalidad |
| Laborales | tipo de contrato, jornada, horas/semana, años de experiencia, meses en desempleo |
| Renta individual | renta neta salarial, renta no monetaria |
| Renta del hogar | renta disponible neta, unidades de consumo, renta per cápita |
| Vivienda | régimen de tenencia, gastos de vivienda, cuota hipoteca, importe alquiler |
| Privación material | puede_vacaciones, puede_proteina_2dias, puede_calefaccion_invierno, puede_sustituir_muebles |
| Salud | estado de salud, enfermedad crónica, limitación de actividad |
| Indicadores INE | hogar_riesgo_pobreza, carencia_material_social_severa, AROPE 2020 y 2030 |

---

## 5. Pipeline end-to-end

### 5.1 Bronze → Silver (`bronze_to_silver.py`)

| Paso | Descripción |
|------|-------------|
| 0. Carga | 4 CSVs originales del INE (D, H, P, R) |
| 1. Joins | D+H (hogar), R+P (persona), hogar+persona |
| 2. Filtros | Comunidad de Madrid + asalariados activos → 2.947 filas |
| 3. Selección de variables | 81 variables con justificación por grupo |
| 4. Imputación de nulos | Usando flags `_F` del INE (grupos A, B y C) |
| 5. Eliminación de flags | Elimina columnas `_F` y auxiliares |
| 6. Rename | Variables originales ECV → snake_case descriptivo |
| 7. Decodificación | Códigos numéricos del INE → etiquetas legibles |
| 8. Construcción del target | `estres_financiero_alto` (binario, ≥2 de 5 condiciones) |
| 9. Feature Engineering | `renta_hogar_per_capita`, `ratio_carga_vivienda`, `precariedad_laboral` |
| 10. Train/Test split | 80/20 estratificado (`random_state=42`) |

**Salidas:** `dataset_analitico.csv`, `train_silver.csv`, `test_silver.csv`

> **Regla metodológica:** el test set permanece completamente oculto durante el EDA y el desarrollo del modelo.

### 5.2 EDA (`01_eda.ipynb`)

Análisis realizado exclusivamente sobre `train_silver.csv` (2.357 filas × 67 columnas).

| Sección | Contenido |
|---------|-----------|
| 1 | Auditoría inicial: tipos, nulos, cardinalidades |
| 2 | Distribución del target y desbalanceo |
| 3–9 | Variables demográficas, laborales, renta, vivienda, privación, salud, dinámica de ingresos, indicadores INE |
| 10 | Correlaciones Pearson y Spearman, multicolinealidad |
| 11 | Análisis de nulos |
| 12 | Selección de features y exportación de listas para el pipeline |

**Features seleccionadas para el pipeline:**
- **15 numéricas:** `renta_hogar_per_capita`, `gastos_vivienda`, `ratio_carga_vivienda`, `ocupacion_isco08`, `renta_neta_salarial`, `importe_alquiler`, `num_habitaciones`, `anios_experiencia`, `meses_desempleo_ref`, `cuota_hipoteca`, `renta_no_monetaria_salarial`, `meses_desempleo_5anios`, `unidades_consumo`, `horas_semana`, `precariedad_laboral`
- **25 categóricas:** `jornada`, `pais_nacimiento`, `hogar_riesgo_pobreza`, `tipo_vivienda`, `arope_2030`, `expectativa_ingresos_12m`, `regimen_tenencia`, `puede_proteina_2dias`, `carencia_material_social_severa`, `puede_sustituir_muebles`, `puede_vacaciones`, `nivel_estudios`, `estado_salud`, `limitacion_actividad`, `hogar_carencia_material`, `arope_2020`, `sexo`, `tipo_contrato`, `cambio_ingresos_12m`, `motivo_disminucion_ingresos`, `motivo_aumento_ingresos`, `baja_intensidad_laboral_2020`, `nacionalidad`, `puede_calefaccion_invierno`, `tipo_hogar`

### 5.3 Modelado (`02_modelado.ipynb` y `model_pipeline.py`)

Preprocesado encapsulado en un `ColumnTransformer` con fit exclusivo sobre train:

| Tipo | Transformación |
|------|---------------|
| Numéricas | `SimpleImputer(median)` → `log1p` en variables de renta → `StandardScaler` |
| Categóricas nominales | `SimpleImputer(most_frequent)` → `OneHotEncoder` |
| Categóricas ordinales | `SimpleImputer(most_frequent)` → `OrdinalEncoder` |

---

## 6. Modelado

### Elección y justificación de las métricas

El problema presenta un **desbalanceo estructural 1:5** (16% de casos positivos), lo que hace que la accuracy sea una métrica engañosa: un clasificador trivial que siempre predice "sin estrés" alcanzaría ~84% de accuracy sin aprender nada.

Por este motivo se establecieron dos métricas complementarias:
- **ROC-AUC**: métrica de ranking que mide la capacidad del modelo para separar las dos clases a través de todos los umbrales posibles. Es insensible al desbalanceo y permite comparar modelos de forma independiente al umbral de decisión elegido.
- **F1-Score (clase 1)** (métrica primaria): media armónica de precisión y recall sobre la clase "estrés alto". Es la métrica que guía tanto la selección del algoritmo como la optimización de hiperparámetros, por dos razones:
  - **Asimetría del coste de error**: en un contexto de RRHH, un falso negativo (empleado con estrés financiero alto que el modelo no detecta) es significativamente más costoso que un falso positivo. El F1 penaliza los fallos en ambas direcciones, pero su orientación hacia la clase minoritaria lo hace más sensible al recall.
  - **Criterio de selección de algoritmo**: en el baseline con validación cruzada (5-fold estratificado), RandomForest obtuvo mejor AUC (0.9529 vs 0.9480 de LightGBM), pero LightGBM fue el mejor en F1 (0.7549 vs 0.6951 de RandomForest). Dado que el F1 es la métrica primaria, LightGBM fue el candidato natural para la optimización fina.

### Proceso de selección

| Etapa | Modelo | ROC-AUC | F1 (clase 1) |
|-------|--------|--------:|-------------:|
| Baseline CV | LightGBM · todas las features | 0.9480 | 0.7549 |
| GridSearchCV | LightGBM · todas las features · params optimizados | 0.9514 | 0.7687 |
| Optuna (NSGA-II, 100 trials) | LightGBM · búsqueda multiobjetivo | 0.9525 | 0.7549 |
| **Modelo final** | **LightGBM · GridSearch · 30 features por importancia** | **0.9590** | **0.8105** |

### Parámetros del modelo final

```python
LGBMClassifier(
    class_weight   = 'balanced',   # compensa el desbalanceo 1:5
    learning_rate  = 0.2,
    max_depth      = 10,
    n_estimators   = 100,
    num_leaves     = 63,
    random_state   = 42,
    n_jobs         = -1
)
```

### Métricas finales sobre test (holdout)

| Métrica | Valor |
|---------|------:|
| ROC-AUC | **0.9590** |
| F1 (estrés alto) | **0.8105** |
| Average Precision | **0.8671** |
| Accuracy global | **0.94** |
| Precision (estrés alto) | 0.79 |
| Recall (estrés alto) | **0.83** |

### Feature importance (grupos por gain)

| Grupo | Features principales | % gain acumulado |
|-------|---------------------|----------------:|
| Privación material | `puede_sustituir_muebles`, `puede_vacaciones` | 40.7% |
| Renta y vivienda | `renta_hogar_per_capita`, `gastos_vivienda`, `renta_neta_hogar`, `ratio_carga_vivienda`, `cuota_hipoteca` | 21.0% |
| Posición laboral | `ocupacion_isco08`, `anios_experiencia`, `renta_neta_salarial` | 7.4% |
| Deuda | `carga_prestamos_no_vivienda` | 5.6% |
| Demografía / hogar | `anio_nacimiento`, `num_habitaciones` | 4.1% |

> Los indicadores de privación **identifican** el riesgo; las variables de renta **gradúan** su intensidad.

---

## 7. Aplicación Streamlit (FinWellsor)

La aplicación tiene **login por rol** y dos perfiles de usuario:

### Rol Empleado
- **Cuestionario:** el empleado responde preguntas sobre su situación financiera, laboral y personal.
- **Predicción:** el modelo devuelve el nivel de riesgo (Alto / Medio / Bajo) junto con una estimación del salario neto por ocupación.
- **Privacidad:** se requiere consentimiento explícito antes de procesar los datos.

### Rol RRHH
- **Contexto:** descripción del proyecto y motivación.
- **Usuarios y Valor:** casos de uso y propuesta de valor para la organización.
- **Análisis EDA:** visualizaciones interactivas de los hallazgos clave.
- **Presupuesto:** análisis de costes y retorno esperado.
- **Limitaciones y Futuro:** hoja de ruta a corto y medio plazo.
- **Respuestas Empleados:** panel con el histórico de predicciones del equipo.

```bash
streamlit run app.py
```

---

## 8. Instalación y ejecución

Este proyecto utiliza **uv** como gestor de entornos y dependencias.

### Requisitos previos
- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) instalado

### Instalación

```bash
# Clonar el repositorio
git clone <url-del-repo>
cd tfm_project

# Crear el entorno e instalar dependencias con uv
uv sync
```

### Ejecución del pipeline completo

```bash
# Ejecuta bronze_to_silver + entrenamiento del modelo
uv run python main.py
```

### Ejecución por pasos

```bash
# Solo limpieza Bronze → Silver
uv run python src/scripts/bronze_to_silver.py

# Solo entrenamiento del modelo
uv run python src/scripts/model_pipeline.py

# Aplicación Streamlit
uv run streamlit run app.py
```

### Notebooks

```bash
uv run jupyter notebook src/notebooks/01_eda.ipynb
uv run jupyter notebook src/notebooks/02_modelado.ipynb
```

> **Nota:** los notebooks deben ejecutarse desde la carpeta `src/` como directorio de trabajo (el código incluye `os.chdir('..')` al inicio de cada notebook para resolver las rutas relativas correctamente).

---

## 9. Dependencias

Gestionadas en `pyproject.toml` con uv:

| Paquete | Uso |
|---------|-----|
| `lightgbm ≥ 4.6` | Modelo de clasificación principal |
| `scikit-learn ≥ 1.8` | Pipeline, preprocesado, métricas |
| `pandas ≥ 3.0` | Manipulación de datos |
| `numpy ≥ 2.4` | Operaciones numéricas |
| `optuna ≥ 4.9` | Optimización de hiperparámetros (notebooks) |
| `xgboost ≥ 3.2` | Algoritmo alternativo evaluado en baseline |
| `streamlit ≥ 1.58` | Interfaz de usuario |
| `matplotlib ≥ 3.10` | Visualizaciones EDA |
| `seaborn ≥ 0.13` | Visualizaciones EDA |
| `plotly ≥ 6.8` | Visualizaciones interactivas en la app |
| `joblib ≥ 1.5` | Serialización auxiliar |

---

## 10. Resultados

El modelo final es un **LightGBM optimizado con GridSearchCV** reentrenado sobre las 30 features más informativas por gain. Detecta correctamente el **83% de los casos de estrés financiero alto**, con una precisión del 79%.

El hallazgo principal del análisis es que el estrés financiero es un **fenómeno multidimensional**: los indicadores de privación percibida y la capacidad de absorber shocks económicos son más predictivos que el nivel de renta puntual. Un asalariado con renta media pero sin capacidad de afrontar imprevistos presenta un perfil de riesgo más elevado que uno con renta baja pero sin deudas y con ahorros.

Esto valida tanto el diseño del target (≥2 de 5 condiciones simultáneas) como la aproximación metodológica adoptada.

---

*TFM · Máster de Data Science · ECV 2025 (INE)*  
*Elaborado por Soraya Malpica Montes*  
***LinkedIn** → https://www.linkedin.com/in/sorayamm/*
