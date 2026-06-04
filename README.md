# Predicción de Estrés Financiero en Asalariados de la Comunidad de Madrid

Modelo de clasificación binaria que predice el riesgo de estrés financiero alto en asalariados de la Comunidad de Madrid, entrenado sobre los microdatos de la **Encuesta de Condiciones de Vida (ECV) 2025** del INE. El sistema genera un score de riesgo individual que puede integrarse en estrategias de bienestar laboral y gestión preventiva de RRHH.

---

## Motivación

El estrés financiero es uno de los principales factores de riesgo para la salud mental y el rendimiento laboral. Sin embargo, su detección en el entorno corporativo sigue siendo reactiva: los departamentos de RRHH actúan cuando el trabajador ya ha manifestado dificultades, no antes.

Disponer de un modelo predictivo permite identificar perfiles de riesgo con antelación, priorizando intervenciones (planes de ahorro, asesoramiento financiero, beneficios flexibles) en la población más vulnerable. En el contexto de la Comunidad de Madrid -economía de alta renta media pero con fuerte presión inmobiliaria y desigualdad creciente- esta capacidad predictiva resulta especialmente relevante.

---

## Pipeline

El proyecto sigue una arquitectura medallón en cuatro etapas:

```
ECV 2025 (INE)
    │
    ▼
bronze_to_silver.py          # Ingesta, limpieza y construcción del target
    │
    ▼
01_eda.ipynb                 # Análisis exploratorio y selección de features
    │
    ▼
02_modelado.ipynb            # Preprocesado, modelado y evaluación
    │
    ▼
modelo_produccion.py         # Inferencia end-to-end sobre nuevos datos
```

### Etapa 1 - Bronze → Silver (`bronze_to_silver.py`)

- Join de los 4 ficheros del INE (D, H, P, R)
- Filtro de población: Comunidad de Madrid + asalariados activos → **n = 2.947**
- Limpieza, decodificación de códigos INE e imputación semántica
- Construcción del target: `estres_financiero_alto` (≥ 2 de 5 condiciones de dificultad)
- Feature engineering determinista: `renta_hogar_per_capita`, `ratio_carga_vivienda`, `precariedad_laboral`, agrupación de `nivel_estudios`
- Train/test split estratificado (80/20, `random_state=42`)
- Salida: `data/02_silver/train_silver.csv` y `data/02_silver/test_silver.csv`

### Etapa 2 - EDA (`01_eda.ipynb`)

- Análisis exclusivamente sobre `train_silver.csv` (evita contaminación del test)
- Correlación de Spearman con el target + test Mann-Whitney por variable
- Detección de multicolinealidad (|ρ| > 0.70)
- Feature importance preliminar con un GBM rápido
- Selección final de features: pool estadístico + Spearman + GBM, descartando redundantes
- Salida: `data/03_gold/raw/FEATURES_NUM.txt` y `FEATURES_CAT.txt`

### Etapa 3 - Modelado (`02_modelado.ipynb`)

- Detección y exclusión de variables con leakage semántico
- Exclusión de variables con un porcentaje alto de nulos, 100% de cardinalidad o con valores constantes.
- Pipeline integrado sklearn: preprocesado + modelo dentro del CV
  - *Numéricas:* `SimpleImputer(median)` → `log1p` sobre rentas → `StandardScaler`
  - *Nominales:* `SimpleImputer(most_frequent)` → `OneHotEncoder`
  - *Ordinales:* `SimpleImputer(most_frequent)` → `OrdinalEncoder` (orden semántico)
- Baseline con `StratifiedKFold(5)`: Logistic Regression, Random Forest, XGBoost, LightGBM
- Modelo seleccionado: **LightGBM** (mejor F1; diferencia mínima de AUC frente a RF)
- Optimización: `GridSearchCV` con `scoring='f1'`
- Salida: `data/04_modelos/pipeline_final.pkl` y `metadata.json`

### Etapa 4 - Producción (`modelo_produccion.py`)

- Carga `pipeline_final.pkl` + `metadata.json`
- Recibe un CSV silver → `predict_proba` → umbral óptimo de decisión
- Exporta `score_riesgo`, `clase_predicha` y `nivel_riesgo` en formato tabulado
- Salida: `data/05_predicciones/predicciones_*.csv`

---

## Dataset

| Atributo | Valor |
|----------|-------|
| **Fuente** | ECV 2025 - Instituto Nacional de Estadística (INE) |
| **Población** | Asalariados en activo, Comunidad de Madrid |
| **n final** | 2.947 observaciones |
| **Split** | 2.358 train / 589 test |
| **Target** | `estres_financiero_alto` |
| **Distribución del target** | Clase 0 (sin estrés): 84.2% / Clase 1 (estrés alto): 15.8% |
| **Ratio de desbalanceo** | 1 : 5.3 |
| **Features finales** | ~64 variables (numéricas, nominales, ordinales) |

El target se define como la concurrencia simultánea de **≥ 2 de 5 condiciones de dificultad financiera del hogar**, capturando vulnerabilidad estructural y no dificultades puntuales de liquidez.

---

## Instalación y uso

### Requisitos

El proyecto usa [uv](https://docs.astral.sh/uv/) como gestor de entorno y dependencias. Las dependencias están fijadas en `pyproject.toml` y `uv.lock`; se exportan también a `requirements.txt` para entornos que no dispongan de uv.

**Con uv (recomendado):**

```bash
# Instalar uv si no está disponible
pip install uv

# Crear el entorno e instalar dependencias en un solo paso
uv sync
```

**Con pip (alternativo):**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Ejecución del pipeline completo

```bash
# 1. Generar datasets silver y split train/test
python bronze_to_silver.py

# 2. Análisis exploratorio y selección de features
#    Ejecutar el notebook en orden secuencial
jupyter notebook 01_eda.ipynb

# 3. Modelado, optimización y evaluación
jupyter notebook 02_modelado.ipynb

# 4. Inferencia sobre nuevos datos
python modelo_produccion.py --input data/02_silver/nuevos_datos.csv
```

### Script de producción

```bash
python modelo_produccion.py --input <ruta_csv_silver> [--output <ruta_salida>]
```

El script espera un CSV con el mismo esquema de columnas que `train_silver.csv`. Genera un archivo de predicciones con tres columnas adicionales: `score_riesgo` (probabilidad continua 0–1), `clase_predicha` (0/1) y `nivel_riesgo` (bajo / medio / alto).

---

## Resultados

### Métricas del modelo final (LightGBM)

| Métrica | CV - train | Test |
|---------|-----------|------|
| ROC-AUC | 0.9514 | **0.9618** |
| F1 clase 1 (umbral 0.5) | 0.7687 | **0.7889** |
| Brecha CV → test | - | +0.010 AUC / +0.020 F1 |

La brecha positiva CV → test es esperada y no indica sobreajuste: el modelo final entrena sobre el 100% del conjunto train frente al ~80% disponible en cada fold de validación cruzada.

### Top 10 features por importancia (LightGBM gain)

| Rank | Feature | Gain | % del total |
|------|---------|------|-------------|
| 1 | `puede_sustituir_muebles_Sí` | 2280.3 | 25.2% |
| 2 | `puede_sustituir_muebles_No` | 787.9 | 8.7% |
| 3 | `puede_vacaciones_Sí` | 618.5 | 6.8% |
| 4 | `renta_hogar_per_capita` | 570.0 | 6.3% |
| 5 | `carga_prestamos_no_vivienda` | 507.1 | 5.6% |
| 6 | `gastos_vivienda` | 473.4 | 5.2% |
| 7 | `renta_neta_hogar` | 378.8 | 4.2% |
| 8 | `ratio_carga_vivienda` | 236.9 | 2.6% |
| 9 | `cuota_hipoteca` | 231.2 | 2.6% |
| 10 | `ocupacion_isco08` | 194.8 | 2.2% |

> **Hallazgo principal:** los indicadores de privación material (`puede_sustituir_muebles`, `puede_vacaciones`) superan a la renta como predictores. La privación es un indicador de *stock* -vulnerabilidad acumulada estructural-; la renta es *flujo* y puede ser puntualmente baja sin reflejar una crisis real. Los indicadores de privación identifican el riesgo en los primeros nodos del árbol; las variables de renta gradúan su intensidad en los nodos más profundos.

18 features presentan gain = 0 (nunca utilizadas): categorías minoritarias de sector CNAE y tipos de hogar unipersonal.

---

## Decisiones metodológicas

| Decisión | Justificación |
|----------|---------------|
| **Target ≥ 2 de 5 condiciones** | Captura vulnerabilidad estructural; evita detectar dificultades puntuales de liquidez |
| **Split antes del preprocesado** | Previene data leakage en imputación estadística, OHE y escalado |
| **Pipeline integrado dentro del CV** | El preprocesado fuera del CV infla el AUC ~2 pp por leakage sutil |
| **Exclusión de indicadores INE derivados** | `arope`, `carencia_material` → AUC = 1.000 por solapamiento directo con el target |
| **`class_weight='balanced'`** | Gestión del desbalanceo 1:5 sin SMOTE (SMOTE dentro del CV introduce leakage) |
| **LightGBM sobre Random Forest** | RF: AUC=0.9529, F1=0.6951 vs LGBM: AUC=0.9480, F1=0.7549 - ventaja de +10 pp en F1 |
| **GridSearch sobre Optuna** | Con n ≈ 2.400 el espacio de hiperparámetros es reducido; Optuna no aporta mejora |
| **Métrica principal: F1 clase 1** | Los falsos negativos (riesgo no detectado) son más costosos que los falsos positivos |
| **Umbral óptimo de decisión** | Con desbalanceo 1:5 el umbral por defecto (0.5) no maximiza F1 |

---

## Stack tecnológico

| Categoría | Librerías |
|-----------|-----------|
| Manipulación de datos | `pandas`, `numpy` |
| Machine Learning | `scikit-learn`, `lightgbm`, `xgboost` |
| Optimización de hiperparámetros | `scikit-learn GridSearchCV`, `optuna` |
| Visualización | `matplotlib`, `seaborn` |
| Serialización | `pickle` |
| Entorno | Python 3.11, Jupyter |

---

## Estructura del repositorio

```
proyecto/
├── src/
│   ├── data/
│   │   ├── 01_raw/              # Ficheros originales INE (D, H, P, R)
│   │   ├── 02_silver/           # Dataset limpio + train/test split
│   │   ├── 03_gold/             # Datasets preprocesados + FEATURES_*.txt
│   │   ├── 04_modelos/          # pipeline_final.pkl + metadata.json
│   │   └── 05_predicciones/     # Outputs del script de producción
│   └── img/                     # Gráficos generados en los notebooks
├── bronze_to_silver.py          # Pipeline Silver: limpieza + split
├── modelo_produccion.py         # Script de inferencia end-to-end
├── 01_eda.ipynb                 # Análisis exploratorio (solo sobre train)
├── 02_modelado.ipynb            # Preprocesado + modelado + evaluación
└── README.md
```

---

## Notas sobre reproducibilidad

- Todo el código usa `random_state=42` donde aplica.
- Los ficheros originales del INE (carpeta `01_raw/`) no se incluyen en el repositorio por política de distribución del INE. Pueden descargarse directamente desde [https://www.ine.es/dyngs/INEbase/es/operacion.htm?c=Estadistica_C&cid=1254736176807&menu=resultados&idp=1254735976608](https://www.ine.es/dyngs/INEbase/es/operacion.htm?c=Estadistica_C&cid=1254736176807&menu=resultados&idp=1254735976608).
- El pipeline es reproducible a partir de los datasets silver incluidos en `02_silver/`.
