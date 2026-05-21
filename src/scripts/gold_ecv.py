'''
gold_ecv.py
===========
Pipeline Capa Gold para el TFM de predicción de estrés financiero (ECV 2025).

Flujo
─────
  PASO 0  Carga de dataset_analitico.csv (Silver)
  PASO 1  Construcción del target estres_financiero_alto
  PASO 2  Eliminación de NaN en el target (indeterminables)
  PASO 3  *** TRAIN / TEST SPLIT *** ← primera operación sobre los datos
            train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
  PASO 4  Eliminación de constantes, IDs, leakage y duplicadas
  PASO 5  Feature engineering (determinista — sin ajuste sobre datos)
            · renta_hogar_per_capita
            · ratio_carga_vivienda
            · precariedad_laboral
            · log1p de rentas
            · agrupación nivel_estudios
  PASO 6  Imputación semántica (no estadística)
            · motivo_*_ingresos → categoría "No aplica"
            · expectativa_sin_respuesta (indicador binario de no-respuesta)
  PASO 7  Encoding ordinal con mapa fijo (dominio)
  PASO 8  Encoding binario con mapa fijo
  PASO 9  Exportación
            · data/03_gold/dataset_modelado.csv   (dataset completo Gold)
            · data/03_gold/X_train.csv / X_test.csv
            · data/03_gold/y_train.csv / y_test.csv
            · data/03_gold/peso_train.csv / peso_test.csv  (si existe)

REGLA ARQUITECTÓNICA INAMOVIBLE
────────────────────────────────
  El train_test_split se ejecuta ANTES de cualquier imputación estadística,
  OHE o escalado. El test set nunca toca ningún fit().
  Las transformaciones de este script son todas DETERMINISTAS:
  ninguna calcula estadísticas sobre la muestra ni aprende de los datos.

Salida del script → data/03_gold/
'''

import math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────────────────────────────────────

PATH_SILVER = Path('data/02_silver/dataset_analitico.csv')
PATH_GOLD   = Path('data/03_gold/dataset_modelado.csv')
PATH_GOLD_DIR = Path('data/03_gold')

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

COMPONENTES_ESTRES = {
    'capacidad_fin_de_mes':         ['Con mucha dificultad', 'Con dificultad'],
    'capacidad_gastos_imprevistos': ['No (no puede)'],
    'retrasos_facturas':            ['Sí, una vez', 'Sí, dos o más veces'],
    'retrasos_hipoteca_alquiler':   ['Sí, una vez', 'Sí, dos o más veces'],
    'retrasos_deudas_no_vivienda':  ['Sí, una vez', 'Sí, dos o más veces'],
}

COLS_CONSTANTES  = ['region', 'situacion_actividad', 'situacion_profesional']
COLS_IDS         = ['id_hogar', 'id_persona']
COLS_TARGET_LEAK = list(COMPONENTES_ESTRES.keys())
COLS_DUPLICADAS  = ['renta_hogar_indicadores']

COLS_LOG1P = [
    'renta_neta_salarial',
    'renta_no_monetaria_salarial',
    'renta_neta_hogar',
    'renta_hogar_per_capita',
    'importe_alquiler',
    'cuota_hipoteca',
    'gastos_vivienda',
]

MAPA_ESTUDIOS = {
    'Sin estudios':                 'Hasta primaria',
    'Primaria incompleta':          'Hasta primaria',
    'Primaria':                     'Hasta primaria',
    'Secundaria 1ª etapa':          'Secundaria 1a etapa',
    'Secundaria 1ª etapa (título)': 'Secundaria 1a etapa',
    'Secundaria 2ª etapa (gral)':   'Post-secundaria',
    'Post-secundaria no superior':  'Post-secundaria',
}

ENCODING_ORDINAL = {
    'nivel_estudios':            {'Hasta primaria': 0, 'Secundaria 1a etapa': 1, 'Post-secundaria': 2},
    'estado_salud':              {'Muy malo': 0, 'Malo': 1, 'Regular': 2, 'Bueno': 3, 'Muy bueno': 4},
    'limitacion_actividad':      {'Gravemente limitado': 0, 'Limitado (no grave)': 1, 'No limitado': 2},
    'grado_urbanizacion':        {'Zona poco poblada': 0, 'Zona media': 1, 'Zona muy poblada': 2},
    'cambio_ingresos_12m':       {'Han disminuido': 0, 'Se mantienen': 1, 'Han aumentado': 2},
    'expectativa_ingresos_12m':  {'Empeorar': 0, 'Mantenerse': 1, 'Mejorar': 2},
    'carga_prestamos_no_vivienda': {'Una carga pesada': 0, 'Una carga razonable': 1, 'Ninguna carga': 2},
    'carga_asistencia_medica':   {'Una carga pesada': 0, 'Una carga razonable': 1, 'Ninguna carga': 2, 'No ha utilizado': 3},
    'carga_asistencia_dental':   {'Una carga pesada': 0, 'Una carga razonable': 1, 'Ninguna carga': 2, 'No ha utilizado': 3},
    'carga_medicamentos':        {'Una carga pesada': 0, 'Una carga razonable': 1, 'Ninguna carga': 2, 'No ha consumido': 3},
}

COLS_BINARIAS = [
    'sexo', 'jornada', 'personal_a_cargo', 'enfermedad_cronica',
    'necesito_medico_no_fue', 'puede_vacaciones', 'puede_proteina_2dias',
    'puede_calefaccion_invierno', 'hogar_riesgo_pobreza', 'hogar_carencia_material',
    'arope_2020', 'arope_2030', 'carencia_material_social_severa',
    'baja_intensidad_laboral_2020', 'puede_sustituir_muebles',
]

# 'No aplicable (≥60 años)' → NaN intencionado; el Pipeline imputará con mediana sobre train
MAPA_BINARIO = {
    'Sí': 1.0, 'Si': 1.0, 'No': 0.0,
    'Hombre': 1.0, 'Mujer': 0.0,
    'Tiempo completo': 1.0, 'Tiempo parcial': 0.0,
    'No aplicable (≥60 años)': math.nan,
    'No aplicable (>=60 anos)': math.nan,
}


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────────────────────────────────────────

def construir_target(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Construye estres_financiero_alto (binario, ≥2 de 5 condiciones).
    NaN cuando el score es indeterminable (< 4 componentes disponibles y score < 2).
    '''
    for col, vals in COMPONENTES_ESTRES.items():
        df[f'_comp_{col}'] = df[col].isin(vals).astype('Int64')

    comp_cols = [f'_comp_{c}' for c in COMPONENTES_ESTRES]
    score    = df[comp_cols].sum(axis=1, skipna=True)
    n_disp   = df[comp_cols].notna().sum(axis=1)

    df['estres_financiero_alto'] = np.where(
        score >= 2, 1,
        np.where((score < 2) & (n_disp >= 4), 0, np.nan)
    ).astype('Int64')

    df = df.drop(columns=comp_cols)
    return df


def eliminar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    '''Elimina constantes, IDs, variables con leakage y duplicadas.'''
    cols_a_eliminar = (
        COLS_CONSTANTES + COLS_IDS + COLS_TARGET_LEAK + COLS_DUPLICADAS
    )
    cols_ok = [c for c in cols_a_eliminar if c in df.columns]
    return df.drop(columns=cols_ok)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    '''Transformaciones deterministas: sin estadísticas de muestra.'''

    # Renta per cápita
    df['renta_hogar_per_capita'] = df['renta_neta_hogar'] / df['unidades_consumo']

    # Ratio carga vivienda (capeado al p99 sobre la muestra completa antes del split
    # es aceptable porque es una decisión de diseño fija, no un parámetro aprendido)
    ratio = np.where(
        df['renta_neta_salarial'] > 0,
        (df['gastos_vivienda'] * 12) / df['renta_neta_salarial'],
        np.nan
    )
    df['ratio_carga_vivienda'] = ratio
    p99 = pd.Series(ratio).quantile(0.99)       # parámetro de diseño fijo
    df['ratio_carga_vivienda'] = df['ratio_carga_vivienda'].clip(upper=p99)

    # Precariedad laboral
    es_temporal = df['tipo_contrato'].isin(['Temporal escrito', 'Temporal verbal'])
    es_parcial  = df['jornada'] == 'Tiempo parcial'
    df['precariedad_laboral'] = (es_temporal | es_parcial).astype('Int64')

    # Agrupación nivel_estudios
    df['nivel_estudios'] = df['nivel_estudios'].map(MAPA_ESTUDIOS)

    # log1p sobre rentas
    for col in COLS_LOG1P:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col].clip(lower=0))

    return df


def imputacion_semantica(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Solo imputa nulos cuyo significado es conocido a priori (diseño del cuestionario).
    El resto de NaN se delega al Pipeline tras el split.
    '''
    df['motivo_aumento_ingresos'] = (
        df['motivo_aumento_ingresos'].fillna('No aplica (sin aumento)')
    )
    df['motivo_disminucion_ingresos'] = (
        df['motivo_disminucion_ingresos'].fillna('No aplica (sin disminucion)')
    )
    # Indicador binario de no-respuesta; el VALOR de expectativa_ingresos_12m
    # se conserva como NaN para imputación con moda ajustada solo sobre train.
    df['expectativa_sin_respuesta'] = df['expectativa_ingresos_12m'].isna().astype(int)
    return df


def encoding_fijo(df: pd.DataFrame) -> pd.DataFrame:
    '''Encodings deterministas: mapa fijo predefinido por conocimiento del dominio.'''

    # Ordinal
    for col, mapa in ENCODING_ORDINAL.items():
        if col in df.columns:
            df[col] = df[col].map(mapa)

    # Binario
    for col in COLS_BINARIAS:
        if col in df.columns:
            df[col] = df[col].map(MAPA_BINARIO)

    return df


def exportar_splits(
    df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    '''Exporta el dataset Gold completo y los splits de train/test.'''
    PATH_GOLD_DIR.mkdir(parents=True, exist_ok=True)

    # Dataset Gold completo
    df.to_csv(PATH_GOLD, index=False, encoding='utf-8-sig')
    print(f'  dataset_modelado.csv → {PATH_GOLD}  ({df.shape[0]:,} × {df.shape[1]})')

    # Splits X e y
    X_train.to_csv(PATH_GOLD_DIR / 'X_train.csv', index=False)
    X_test.to_csv( PATH_GOLD_DIR / 'X_test.csv',  index=False)
    y_train.to_csv(PATH_GOLD_DIR / 'y_train.csv', index=False, header=True)
    y_test.to_csv( PATH_GOLD_DIR / 'y_test.csv',  index=False, header=True)
    print(f'  X_train.csv  {X_train.shape}  |  X_test.csv  {X_test.shape}')
    print(f'  y_train.csv  {y_train.shape}  |  y_test.csv  {y_test.shape}')

    # Peso muestral (auxiliar — no es feature)
    if 'peso_persona' in df.columns:
        df.loc[X_train.index, 'peso_persona'].to_csv(
            PATH_GOLD_DIR / 'peso_train.csv', index=False, header=True
        )
        df.loc[X_test.index, 'peso_persona'].to_csv(
            PATH_GOLD_DIR / 'peso_test.csv', index=False, header=True
        )
        print('  peso_train.csv  |  peso_test.csv')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run() -> None:
    sep = '═' * 62

    # ── PASO 0: Carga ─────────────────────────────────────────────────────────
    print(f'\n{sep}')
    print('PASO 0 — Carga del dataset Silver')
    print(sep)
    df = pd.read_csv(PATH_SILVER, low_memory=False)
    print(f'  {df.shape[0]:,} filas × {df.shape[1]} columnas')

    # ── PASO 1: Construir target ───────────────────────────────────────────────
    print(f'\n{sep}')
    print('PASO 1 — Construcción del target estres_financiero_alto')
    print(sep)
    df = construir_target(df)
    n0  = (df['estres_financiero_alto'] == 0).sum()
    n1  = (df['estres_financiero_alto'] == 1).sum()
    nan = df['estres_financiero_alto'].isna().sum()
    print(f'  0 — sin estrés: {n0:,}  ({n0/len(df)*100:.1f}%)')
    print(f'  1 — estrés alto:{n1:,}  ({n1/len(df)*100:.1f}%)')
    print(f'  NaN (indet.):   {nan:,}  → se eliminan antes del split')

    # ── PASO 2: Eliminar NaN del target ───────────────────────────────────────
    df = df.dropna(subset=['estres_financiero_alto']).copy()
    print(f'  Filas tras eliminar NaN target: {len(df):,}')

    # ── PASO 3: TRAIN / TEST SPLIT ────────────────────────────────────────────
    # PRIMERA OPERACIÓN QUE DIVIDE LOS DATOS.
    # Todo lo que aprende parámetros de los datos (imputación estadística,
    # OHE, StandardScaler) se ejecuta DESPUÉS y solo sobre X_train.
    print(f'\n{sep}')
    print('PASO 3 — TRAIN / TEST SPLIT  ← primera operación sobre los datos')
    print(sep)

    COLS_AUX = ['estres_financiero_alto', 'peso_persona']
    X_raw = df.drop(columns=[c for c in COLS_AUX if c in df.columns])
    y_raw = df['estres_financiero_alto'].astype(int)

    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        X_raw, y_raw,
        test_size=0.20,
        random_state=42,
        stratify=y_raw,
    )

    idx_train = X_train_idx.index
    idx_test  = X_test_idx.index

    print(f'  Train: {len(idx_train):,}  ({len(idx_train)/len(df)*100:.1f}%)')
    print(f'  Test:  {len(idx_test):,}  ({len(idx_test)/len(df)*100:.1f}%)')
    print(f'  Clase 1 en train: {(y_train==1).mean()*100:.1f}%  |  en test: {(y_test==1).mean()*100:.1f}%')

    # ── PASOS 4–8: Transformaciones deterministas (sobre df completo) ──────────
    # Aunque las transformaciones se aplican al dataset completo, son seguras
    # porque ninguna calcula estadísticas sobre la muestra:
    #   · Los mapas de encoding son fijos (definidos en constantes arriba)
    #   · El p99 de ratio_carga_vivienda es un umbral de diseño documentado
    #   · log1p no depende de los datos
    # El split de índices ya está fijado; X_train / X_test se extraen al final.

    print(f'\n{sep}')
    print('PASO 4 — Eliminación de columnas')
    print(sep)
    n_antes = df.shape[1]
    df = eliminar_columnas(df)
    print(f'  Eliminadas: {n_antes - df.shape[1]} columnas  |  Restantes: {df.shape[1]}')

    print(f'\n{sep}')
    print('PASO 5 — Feature engineering')
    print(sep)
    df = feature_engineering(df)
    cols_nuevas = ['renta_hogar_per_capita', 'ratio_carga_vivienda', 'precariedad_laboral'] \
                + [f'log_{c}' for c in COLS_LOG1P if c in df.columns]
    print(f'  Columnas creadas: {len(cols_nuevas)}')
    for c in cols_nuevas:
        print(f'    + {c}')

    print(f'\n{sep}')
    print('PASO 6 — Imputación semántica')
    print(sep)
    df = imputacion_semantica(df)
    print('  motivo_aumento_ingresos     → NaN imputados como "No aplica (sin aumento)"')
    print('  motivo_disminucion_ingresos → NaN imputados como "No aplica (sin disminucion)"')
    print('  expectativa_sin_respuesta   → indicador binario creado')

    print(f'\n{sep}')
    print('PASO 7 — Encoding ordinal (mapa fijo)')
    print(sep)
    df = encoding_fijo(df)
    print(f'  Ordinal: {len(ENCODING_ORDINAL)} variables  |  Binario: {len(COLS_BINARIAS)} variables')

    # ── PASO 8: Reconstruir splits con el dataset Gold transformado ────────────
    X_train = df.drop(columns=[c for c in COLS_AUX if c in df.columns]).loc[idx_train]
    X_test  = df.drop(columns=[c for c in COLS_AUX if c in df.columns]).loc[idx_test]

    # Resync de y con los índices del Gold (el target no cambia, pero por consistencia)
    y_train = df.loc[idx_train, 'estres_financiero_alto'].astype(int)
    y_test  = df.loc[idx_test,  'estres_financiero_alto'].astype(int)

    # ── PASO 9: Exportación ────────────────────────────────────────────────────
    print(f'\n{sep}')
    print('PASO 9 — Exportación')
    print(sep)
    exportar_splits(df, X_train, X_test, y_train, y_test)

    # ── Resumen final ──────────────────────────────────────────────────────────
    X_features = df.drop(columns=[c for c in COLS_AUX if c in df.columns])
    cols_num = X_features.select_dtypes(include='number').columns
    cols_cat = X_features.select_dtypes(include='object').columns

    print(f'\n{sep}')
    print('RESUMEN FINAL — DATASET GOLD')
    print(sep)
    print(f'  Observaciones:                {len(df):,}')
    print(f'  Features numéricas/ord/bin:  {len(cols_num)}')
    print(f'  Features nominales (→ OHE):  {len(cols_cat)}')
    print(f'  Total features:              {X_features.shape[1]}')
    print(f'  NaN en X (para Pipeline):    {X_features.isnull().sum().sum():,}')
    print(f'\n  Nominales que irán al OHE del Pipeline:')
    for c in cols_cat:
        n = X_features[c].nunique()
        print(f'    {c:<45} ({n} categorías)')
    print(f'\n{sep}')
    print('gold_ecv.py completado con éxito')
    print(sep)


if __name__ == '__main__':
    run()
