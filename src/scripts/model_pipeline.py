'''
modelo_produccion.py
====================
Script end-to-end de producción para el Modelo 1 (estrés financiero — ECV 2025).

Carga pipeline_final.pkl (preprocesado + LightGBM ya entrenados) y ejecuta
la predicción sobre nuevos datos sin reentrenar nada.

Flujo
─────
  1. Carga metadata.json   → modelo, umbral óptimo, parámetros
  2. Carga pipeline_final.pkl → preprocesado + LightGBM ajustados sobre train
  3. Lee el CSV de entrada
  4. pipeline.transform() → nunca .fit()
  5. Predice: probabilidad + clase binaria con el umbral óptimo
  6. Exporta CSV con score_riesgo, clase predicha y nivel_riesgo
'''

import os
import pickle
import warnings
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.parent.parent)

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
)

from src.utils.constants_utils import PATH_TRAIN_SILVER, PATH_TEST_SILVER, COLS_LOG1P
from src.utils.mapeo_utils import ORDINAL_VARS

from lightgbm import LGBMClassifier


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
TARGET = 'estres_financiero_alto'
SEP    = '═' * 62
RANDOM_STATE = 42
MEJOR_MODELO = 'LightGBM'

# FEATURES_DEL = ['motivo_aumento_ingresos', 'motivo_disminucion_ingresos', 
#                 'id_hogar', 'id_persona', 'region', 'capacidad_fin_de_mes', 
#                 'capacidad_gastos_imprevistos', 'retrasos_facturas', 
#                 'retrasos_hipoteca_alquiler', 'retrasos_deudas_no_vivienda'
#                 ]

FEATURES_IMPORT = [
    "edad", "nivel_estudios", "tipo_contrato", "jornada", "anios_experiencia",
    "renta_neta_salarial", "regimen_tenencia", "cuota_hipoteca", "importe_alquiler",
    "gastos_vivienda", "renta_neta_hogar", "num_miembros_hogar",
    "carga_prestamos_no_vivienda", "tipo_hogar", "puede_proteina_2dias",
    "puede_vacaciones", "puede_sustituir_muebles", "puede_calefaccion_invierno",
    "ratio_carga_vivienda", "ocupacion_isco08", "renta_hogar_per_capita",
    "num_habitaciones", "carga_asistencia_dental", "carga_medicamentos",
    "hogar_carencia_material", "cambio_ingresos_12m", "unidades_consumo",
    "renta_no_monetaria_salarial", "horas_semana", "expectativa_ingresos_12m",
]

PARAMS_GS_F1 = {
    'learning_rate': 0.2,
    'max_depth':     10,
    'n_estimators':  100,
    'num_leaves':    63,
}

PATH_MODELS = 'src/models/'

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────────────────────────────────────────

def carga_datos_silver():
    train = pd.read_csv(PATH_TRAIN_SILVER, low_memory=False)
    test = pd.read_csv(PATH_TEST_SILVER,  low_memory=False)
    return train, test


def log1p_rentas(X):
    X = X.copy().astype(float)
    X[:, IDX_LOG] = np.log1p(np.clip(X[:, IDX_LOG], 0, None))
    return X


def construir_preprocessor(cols_num, cols_cat):
    global IDX_LOG
    IDX_LOG = [cols_num.index(c) for c in COLS_LOG1P if c in cols_num]

    cols_ord = [c for c in cols_cat if c in ORDINAL_VARS]
    cols_nom = [c for c in cols_cat if c not in ORDINAL_VARS]

    pipe_num = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('log1p',   FunctionTransformer(log1p_rentas)),
        ('scaler',  StandardScaler()),
    ])
    pipe_nom = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe',     OneHotEncoder(handle_unknown='ignore',
                                  sparse_output=False, drop='if_binary')),
    ])
    ord_cats = [ORDINAL_VARS[c] for c in cols_ord if c in ORDINAL_VARS]
    pipe_ord = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('enc',     OrdinalEncoder(categories=ord_cats,
                                   handle_unknown='use_encoded_value',
                                   unknown_value=-1)),
    ])
    transformers = []
    if cols_num: transformers.append(('num', pipe_num, cols_num))
    if cols_nom: transformers.append(('nom', pipe_nom, cols_nom))
    if cols_ord: transformers.append(('ord', pipe_ord, cols_ord))
    return ColumnTransformer(transformers=transformers, remainder='drop',
                             verbose_feature_names_out=True)


def construir_pipeline_completo(modelo, cols_num, cols_cat):
    '''
    Encapsula preprocesado + modelo en un único Pipeline.
    Al pasarlo a cross_validate / GridSearchCV, cada fold ajusta
    el preprocesado solo sobre sus datos de train.
    '''
    return Pipeline([
        ('preprocesado', construir_preprocessor(cols_num, cols_cat)),
        ('modelo',       modelo),
    ])


def evaluacion_test(pipeline_final, X_test, y):
    y_prob = pipeline_final.predict_proba(X_test)[:, 1]
    y_pred = pipeline_final.predict(X_test)

    AUC_TEST = roc_auc_score(y, y_prob)
    F1_TEST  = f1_score(y, y_pred)
    AP_TEST  = average_precision_score(y, y_prob)

    print(f'ROC-AUC test:          {AUC_TEST:.4f}')
    print(f'F1 clase 1 test:       {F1_TEST:.4f}')
    print(f'Average Precision:     {AP_TEST:.4f}')

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run():
    print('Ejecutando entrenamiento...')
    print('\n',SEP)
    print('PASO 0: CARGA DE TRAIN Y TEST')
    print(SEP)

    print('\nCargando train_set...')
    print('\nCargando test_set...')
    train, test = carga_datos_silver()
    print('Carga completada')

    # FEATURES_DEL.append(TARGET)
    X_train_import = train[FEATURES_IMPORT]    
    y_train = train[TARGET].astype(int)
    X_test_import  = test[FEATURES_IMPORT]
    y_test = test[TARGET].astype(int)

    print(f'\nTrain: {X_train_import.shape}  |  Test: {X_test_import.shape}')
    print(f'Clase 1 en train: {y_train.mean()*100:.1f}%  |  en test: {y_test.mean()*100:.1f}%')

    FEATURES_NUM = [col for col in FEATURES_IMPORT if X_train_import[col].dtypes in [int,float]]
    FEATURES_CAT = [col for col in X_train_import if col not in FEATURES_IMPORT]

    print('\n',SEP)
    print('PASO 1: ENTRENAMIENTO')
    print(SEP)

    print('''\n
    Entrenando con LightGBM optimizado con Gridsearch.

    Mejores parámetros:
          - 'learning_rate': 0.2,
          - 'max_depth':     10,
          - 'n_estimators':  100,
          - 'num_leaves':    63
          - 'class_weight': 'balanced'
        
    Métricas esperadas en test:
          - ROC-AUC:          0.9590
          - F1 (estrés alto): 0.8105
          - Avg Precision:    0.8671
    ''')

    print('\nEntrenando...')

    lg = LGBMClassifier(
        **PARAMS_GS_F1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    
    lg_import = construir_pipeline_completo(
        lg, FEATURES_NUM, FEATURES_CAT
    )
    lg_import.fit(X_train_import, y_train)
    
    print('Entrenamiento finalizado')

    print(f'\n{SEP}')
    print('PASO 2: EVALUACIÓN CONTRA TEST')
    print(SEP)

    print('\nEvaluando sobre test...')
    evaluacion_test(lg_import, X_test_import, y_test)
    print('Evaluación finalizada')

    print(f'\n{SEP}')
    print('PASO 3: EXPORTACIÓN DEL MODELO')
    print(SEP)

    print('\nGuardando modelo...')
    with open(PATH_MODELS + 'lg_optimo.pkl', 'wb') as f:
        pickle.dump(lg_import, f)
    print('Modelo guardado con éxito')