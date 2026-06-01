'''
01_broze_to_silver.py
===============
Script de limpieza y preparación de los microdatos ECV 2025 para el TFM.
Parte de los CSVs originales del INE (sin mapear).

Flujo:
  1. Carga los 4 ficheros originales
  2. Joins: D+H (hogar), R+P (persona), hogar+persona
  3. Filtro: Comunidad de Madrid + asalariados activos
  4. Selección de variables relevantes (con justificación en comentarios)
  5. Imputación de nulos usando flags _F del INE
  6. Eliminación de flags _F y columnas auxiliares
  7. Rename a snake_case
  8. Decodificación de valores categóricos
  9. Sustitución de códigos de nulo residuales del INE → NaN
 10. Exportación del dataset limpio

Salida: data/ECV_2025/dataset_analitico.csv
'''

# import re
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split

from pathlib import Path
os.chdir(Path(__file__).resolve().parent.parent.parent)

import warnings
warnings.filterwarnings('ignore')

from src.utils.constants_utils import *
from src.utils.functions_utils import cargar_csv
from src.utils.mapeo_utils import *


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────────────────────────────────────────

def construir_target(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Construye estres_financiero_alto (binario, ≥2 de 5 condiciones).
    NaN cuando el score_comp es indeterminable (< 4 componentes disponibles y score_comp < 2).
    '''
    for col, vals in COMPONENTES_ESTRES.items():
        df[f'_comp_{col}'] = df[col].isin(vals).astype('Int64')

    comp_cols = [f'_comp_{c}' for c in COMPONENTES_ESTRES]
    score_comp = df[comp_cols].sum(axis=1, skipna=True)
    not_nulls = df[comp_cols].notna().sum(axis=1)

    df['estres_financiero_alto'] = np.where(
        score_comp >= 2, 1,
        np.where((score_comp < 2) & (not_nulls >= 4), 0, np.nan)
    ).astype(int)

    df = df.drop(columns=comp_cols)
    return df


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
    df['precariedad_laboral'] = (es_temporal | es_parcial).astype(int)

    # Agrupación nivel_estudios
    df['nivel_estudios'] = df['nivel_estudios'].map(MAPA_ESTUDIOS)
    
    return df

def run():
    print("Ejecutando limpieza...")
    sep = '═' * 62


    # PASO 0: CARGA
    print(f'\n{sep}')
    print('CARGA DE DATOS')
    print(sep)

    print('\nCargando ficheros...')
    df_d = cargar_csv('01_datos_hogar.csv')       # Fichero D: datos básicos hogar
    df_h = cargar_csv('02_detalles_hogar.csv')    # Fichero H: detalle hogar (renta, vivienda, bienestar)
    df_p = cargar_csv('03_detalles_adulto.csv')   # Fichero P: datos adultos (trabajo, educación, salud, renta individual)
    df_r = cargar_csv('04_datos_persona.csv')     # Fichero R: datos básicos persona (demográficos + vars derivadas)

    print(f'  D: {df_d.shape} | H: {df_h.shape} | P: {df_p.shape} | R: {df_r.shape}')

    # ══════════════════════════════════════════════════════════════════════════════
    # PASO 1: JOINS
    # ══════════════════════════════════════════════════════════════════════════════
    print(f'\n{sep}')
    print('JOINS DE LOS FICHEROS')
    print(sep)
    # ID hogar: DB030 (D) = HB030 (H)
    df_hogar = df_d.merge(df_h, left_on='DB030', right_on='HB030', how='inner')

    # ID persona: los 6 primeros dígitos de PB030/RB030 identifican el hogar (= DB030)
    # PB030 y RB030 son el mismo ID de persona → join directo entre R y P
    df_persona = df_r.merge(df_p, left_on='RB030', right_on='PB030', how='inner')

    # Extraer ID hogar desde ID persona (dividir por 100 para obtener los primeros dígitos)
    df_persona['id_hogar_join'] = df_persona['RB030'].astype(int) // 100

    # Join hogar + persona
    df = df_hogar.merge(df_persona, left_on='DB030', right_on='id_hogar_join', how='inner')
    print(f'\n  Tras joins: {df.shape}')
    # Output -> Tras joins: (60825, 457)

    # ══════════════════════════════════════════════════════════════════════════════
    # PASO 2: FILTROS
    # ══════════════════════════════════════════════════════════════════════════════
    print(f'\n{sep}')
    print('FILTRO POR COMUNIDAD DE MADRID Y ASALARIADOS')
    print(sep)

    # Comunidad de Madrid
    df = df[df['DB040'].astype(str).str.strip() == 'ES30'].copy()
    print(f'\n  Tras filtro Madrid: {df.shape}')
    # Output -> Tras filtro Madrid: (6035, 457)


    # Asalariados activos: PL032 == 1 (trabajando) + PL040A == 3 (asalariado, empleo actual)
    # PL040A = situación profesional empleo ACTUAL (para quien trabaja)
    # Las columnas vienen como string en los CSVs originales del INE
    df = df[
        (df['PL032'].astype(str).str.strip() == '1') &
        (df['PL040A'].astype(str).str.strip() == '3')
    ].copy()
    print(f'\n  Tras filtro asalariados: {df.shape}')
    # Output -> Tras filtro asalariados: (2947, 457)

    # ══════════════════════════════════════════════════════════════════════════════
    # PASO 3: SELECCIÓN DE VARIABLES
    # Criterio general:
    #   - Se incluyen variables con valor predictivo para estrés financiero
    #   - Se excluyen variables administrativas/técnicas (año, país, ID internos)
    #   - Se excluyen flags _F ahora (se usan abajo para imputar nulos, luego se eliminan)
    #   - Variables de renta bruta se excluyen cuando ya existe la neta (evitar redundancia)
    #   - Variables de meses en estados específicos (PL211*) se excluyen: muy granulares
    #     y con alta correlación entre sí → resumen suficiente con PL032 y PL080
    # ══════════════════════════════════════════════════════════════════════════════
    print(f'\n{sep}')
    print('SELECCIÓN DE VARIABLES')
    print(sep)
    # Varible VARS_SELECCIONADAS en mapeo_utils.py

    # Mantener solo las que existen en el df
    vars_seleccionadas = [v for v in VARS_SELECCIONADAS if v in df.columns]
    df = df[vars_seleccionadas].copy()
    print(f'\n\tTras selección de variables: {df.shape}')
    # Output -> Tras selección de variables: (2947, 81)

    # ══════════════════════════════════════════════════════════════════════════════
    # PASO 4: IMPUTACIÓN DE NULOS USANDO FLAGS _F DEL INE
    #    Flag = 1  → Dato recogido correctamente
    #    Flag = -1 → 'No consta' (respuesta ausente) → NaN en la variable correspondiente
    #    Flag = -6 → 'No recogido por diseño muestral' → NaN en la variable correspondiente
    #    Flag = -2 → 'No aplicable' → depende: valor semántico o NaN estructural
    # ══════════════════════════════════════════════════════════════════════════════
    print(f'\n{sep}')
    print('IMPUTACIÓN DE NULOS USANDO FLAGS _F DEL INE')
    print(sep)
    #-------------------------------------------------
    # Grupo A — flag -2 → valor semántico concreto
    #-------------------------------------------------

    print('\nImputando nulos del Grupo A...')
    # 'HH060', 'cuotahip' y 'PL271'necesitan un tratamiento especial ya que hay que hacer, previamente, una serie de transformaciones antes de la decodificación pertinente
    df = df.loc[:, ~df.columns.duplicated()] 
    for var, (flag, valor) in GRUPO_A.items():
        if flag in df.columns and var in df.columns:
            mask = df[flag] == -2
            n = mask.sum()
            if var in ['HH060', 'cuotahip', 'PL271']:
                df[var] = df[var].str.strip().replace('', np.nan).astype(float)
            if n > 0:
                df.loc[mask, var] = valor

    #-------------------------------------------------
    # Grupo B — flag -2 → NaN estructural
    #-------------------------------------------------
    
    print('Imputando nulos del Grupo B...')
    for var, flag in GRUPO_B.items():
        if flag in df.columns and var in df.columns:
            mask = df[flag] == -2
            df.loc[mask, var] = np.nan

    #-------------------------------------------------
    # **Grupo C — flag -1 (y -6 en PL060) → NaN
    #-------------------------------------------------

    print('Imputando nulos del Grupo C...')
    for var in df.columns:
        if var != var.endswith('_F'):
            for valor in df[var].unique():
                if valor == ' ':
                    mask = df[var] == ' '
                    n = mask.sum()
                    if n > 0:
                        df.loc[mask, var] = np.nan

    if 'PL060_F' in df.columns and 'PL060' in df.columns:
        for codigo, motivo in [(-1, '-1 No consta'), (-6, '-6 No recogido')]:
            mask = df['PL060_F'] == codigo
            n = mask.sum()
            if n > 0:
                df.loc[mask, 'PL060'] = np.nan

    print('Imputaciones finalizadas.')

    # ══════════════════════════════════════════════════════════════════════════════
    # PASO 5: ELIMINAR FLAGS _F Y COLUMNAS AUXILIARES
    # ══════════════════════════════════════════════════════════════════════════════

    print(f'\n{sep}')
    print('ELIMINAR FLAGS _F Y COLUMNAS AUXILIARES')
    print(sep)

    print('\nEliminando Flags...')

    cols_eliminar = [c for c in df.columns if c.endswith('_F')] + ['id_hogar_join']
    df = df.drop(columns=[c for c in cols_eliminar if c in df.columns])

    print('Flags eliminadas.')

    # ══════════════════════════════════════════════════════════════════════════════
    # PASO 6: RENAME A SNAKE_CASE
    # ══════════════════════════════════════════════════════════════════════════════

    print(f'\n{sep}')
    print('RENAME A SNAKE_CASE')
    print(sep)

    print('\nRenombrando las variables...')

    # Variable RENAME_MAP en mapeo_utils.py
    df = df.rename(columns=RENAME_MAP)

    print('Variables renombradas correctamente.')

    # ══════════════════════════════════════════════════════════════════════════════
    # PASO 7: DECODIFICACIÓN DE VALORES CATEGÓRICOS
    #    Los CSVs del INE vienen con todas las columnas como string.
    #    Variables continuas (rentas, horas, etc.) se convierten a numérico sin decodificar.
    # ══════════════════════════════════════════════════════════════════════════════

    print(f'\n{sep}')
    print('DECODIFICACIÓN DE VALORES CATEGÓRICOS')
    print(sep)

    # Convertir a numérico las variables continuas que vienen como string
    cols_numericas_str = [
    'horas_semana', 'anios_experiencia', 'meses_desempleo_ref', 
    # 'meses_desempleo_5anios','renta_neta_salarial', 'renta_no_monetaria_salarial', 'renta_neta_hogar', 'renta_hogar_indicadores', 'unidades_consumo', 'num_miembros_hogar', 'importe_alquiler', 'cuota_hipoteca',
    'ocupacion_isco08', 'num_habitaciones', 'gastos_vivienda'
    ]
    for col in cols_numericas_str:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # DECODIFICACIONES (en mapeo_utils.py): todas las claves como string (formato real del INE)

    print('\nDecodificando...')

    for col, mapping in DECODIFICACIONES.items():
        if col in df.columns:
            # Convertir float a int antes de hacer str, para evitar '1.0' en lugar de '1'
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].where(df[col].isna(), df[col].astype('Int64').astype(str))
            else:
                df[col] = df[col].astype(str).where(df[col].notna(), other=np.nan)
            df[col] = df[col].map(mapping)
        
    print('Variables decodificadas.')

    # ══════════════════════════════════════════════════════════════════════════════
    # PASO 8: CONTRUIR TARGET
    # ══════════════════════════════════════════════════════════════════════════════
    sep = '═' * 62

    print(f'\n{sep}')
    print('CONSTRUCCIÓN TARGET estres_financiero_alto')
    print(sep)

    df = construir_target(df)

    print('\nConstrucción del Target finalizado')

    # ══════════════════════════════════════════════════════════════════════════════
    # PASO 9: FEATURE ENGINEERING
    # ══════════════════════════════════════════════════════════════════════════════

    print(f'\n{sep}')
    print('FEATURE ENGINEERING')
    print(sep)
    df = feature_engineering(df)
    cols_nuevas = ['renta_hogar_per_capita', 'ratio_carga_vivienda', 'precariedad_laboral']
                # + [f'log_{c}' for c in COLS_LOG1P if c in df.columns] INCORPORAR EN EL PIPELINE
    print(f'  Columnas creadas: {len(cols_nuevas)}')
    for c in cols_nuevas:
        print(f'    + {c}')

    print('\nFeature Engineering finalizado')

    print('\nGuardado como dataset_analitico.csv')

    df.to_csv(PATH_SILVER_ANALITICO, index=False)

    # ══════════════════════════════════════════════════════════════════════════════
    # PASO 10: TRAIN / TEST SPLIT
    # ══════════════════════════════════════════════════════════════════════════════

    print(f'\n{sep}')
    print('TRAIN / TEST SPLIT')
    print(sep)

    X_raw = df.drop(columns=[c for c in COLS_AUX if c in df.columns])
    y_raw = df['estres_financiero_alto'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw,
        test_size=0.20,
        random_state=42,
        stratify=y_raw,
    )

    train_set = X_train
    train_set['estres_financiero_alto'] = y_train

    test_set = X_test
    test_set['estres_financiero_alto'] = y_test

    print('\nSplit de TRAIN y TEST finalizado.')

    print('\nGuardado como train_silver.csv y test_silver.csv')

    train_set.to_csv(PATH_TRAIN_SILVER, index=False)
    test_set.to_csv(PATH_TEST_SILVER, index=False)