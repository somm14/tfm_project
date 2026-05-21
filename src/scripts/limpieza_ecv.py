'''
limpieza_ecv.py
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

import re
import pandas as pd
import numpy as np

from pathlib import Path

from src.utils.constants_var import *
from src.utils.cleaning_utils import cargar_csv
from src.utils.mapeo_utils import *

def run():
    print("Ejecutando limpieza...")

    # ══════════════════════════════════════════════════════════════════════════════
    # 1. CARGA
    # ══════════════════════════════════════════════════════════════════════════════


    print('Cargando ficheros...')
    df_d = cargar_csv('01_datos_hogar.csv')       # Fichero D: datos básicos hogar
    df_h = cargar_csv('02_detalles_hogar.csv')    # Fichero H: detalle hogar (renta, vivienda, bienestar)
    df_p = cargar_csv('03_detalles_adulto.csv')   # Fichero P: datos adultos (trabajo, educación, salud, renta individual)
    df_r = cargar_csv('04_datos_persona.csv')     # Fichero R: datos básicos persona (demográficos + vars derivadas)

    print(f'  D: {df_d.shape} | H: {df_h.shape} | P: {df_p.shape} | R: {df_r.shape}')


    # ══════════════════════════════════════════════════════════════════════════════
    # 2. JOINS
    # ══════════════════════════════════════════════════════════════════════════════

    # ID hogar: DB030 (D) = HB030 (H)
    df_hogar = df_d.merge(df_h, left_on='DB030', right_on='HB030', how='inner')

    # ID persona: los 6 primeros dígitos de PB030/RB030 identifican el hogar (= DB030)
    # PB030 y RB030 son el mismo ID de persona → join directo entre R y P
    df_persona = df_r.merge(df_p, left_on='RB030', right_on='PB030', how='inner')

    # Extraer ID hogar desde ID persona (dividir por 100 para obtener los primeros dígitos)
    df_persona['id_hogar_join'] = df_persona['RB030'].astype(int) // 100

    # Join hogar + persona
    df = df_hogar.merge(df_persona, left_on='DB030', right_on='id_hogar_join', how='inner')
    print(f'  Tras joins: {df.shape}')
    # Output -> Tras joins: (60825, 457)


    # ══════════════════════════════════════════════════════════════════════════════
    # 3. FILTROS
    # ══════════════════════════════════════════════════════════════════════════════

    # Comunidad de Madrid
    df = df[df['DB040'].astype(str).str.strip() == 'ES30'].copy()
    print(f'  Tras filtro Madrid: {df.shape}')
    # Output -> Tras filtro Madrid: (6035, 457)


    # Asalariados activos: PL032 == 1 (trabajando) + PL040A == 3 (asalariado, empleo actual)
    # PL040A = situación profesional empleo ACTUAL (para quien trabaja)
    # Las columnas vienen como string en los CSVs originales del INE
    df = df[
        (df['PL032'].astype(str).str.strip() == '1') &
        (df['PL040A'].astype(str).str.strip() == '3')
    ].copy()
    print(f'  Tras filtro asalariados: {df.shape}')
    # Output -> Tras filtro asalariados: (2947, 457)


    # ══════════════════════════════════════════════════════════════════════════════
    # 4. SELECCIÓN DE VARIABLES
    # ══════════════════════════════════════════════════════════════════════════════
    # Criterio general:
    #   - Se incluyen variables con valor predictivo para estrés financiero
    #   - Se excluyen variables administrativas/técnicas (año, país, ID internos)
    #   - Se excluyen flags _F ahora (se usan abajo para imputar nulos, luego se eliminan)
    #   - Variables de renta bruta se excluyen cuando ya existe la neta (evitar redundancia)
    #   - Variables de meses en estados específicos (PL211*) se excluyen: muy granulares
    #     y con alta correlación entre sí → resumen suficiente con PL032 y PL080

    # Varible VARS_SELECCIONADAS en mapeo_utils.py

    # Mantener solo las que existen en el df
    vars_seleccionadas = [v for v in VARS_SELECCIONADAS if v in df.columns]
    df = df[vars_seleccionadas].copy()
    print(f'\tTras selección de variables: {df.shape}')
    # Output -> Tras selección de variables: (2947, 81)


    # ══════════════════════════════════════════════════════════════════════════════
    # 5. IMPUTACIÓN DE NULOS USANDO FLAGS _F DEL INE
    #    Flag = 1  → Dato recogido correctamente
    #    Flag = -1 → 'No consta' (respuesta ausente) → NaN en la variable correspondiente
    #    Flag = -6 → 'No recogido por diseño muestral' → NaN en la variable correspondiente
    #    Flag = -2 → 'No aplicable' → depende: valor semántico o NaN estructural
    # ══════════════════════════════════════════════════════════════════════════════

    #-------------------------------------------------
    # Grupo A — flag -2 → valor semántico concreto
    #-------------------------------------------------
    grupo_a = {
        'HS011':    ('HS011_F',    '3'),  # Sin hipoteca/alquiler → sin retrasos
        'HS021':    ('HS021_F',    '3'),  # Sin facturas → sin retrasos
        'HS031':    ('HS031_F',    '3'),  # Sin deudas → sin retrasos
        'HS150':    ('HS150_F',    '3'),  # Sin préstamos → ninguna carga
        'HH060':    ('HH060_F',      0),  # No alquila → importe = 0
        'cuotahip': ('cuotahip_F',   0),  # Sin hipoteca → cuota = 0
        'HS200':    ('HS200_F',    '4'),  # No usó asistencia médica
        'HS210':    ('HS210_F',    '4'),  # No usó asistencia dental
        'HS220':    ('HS220_F',    '4'),  # No consumió medicamentos
        'PL271':    ('PL271_F',      0),  # 0 meses desempleo últimos 5 años
        'PH040':    ('PH040_F',    '2')  # No necesitó médico
    }

    # 'HH060', 'cuotahip' y 'PL271'necesitan un tratamiento especial ya que hay que hacer, previamente, una serie de transformaciones antes de la decodificación pertinente
    df = df.loc[:, ~df.columns.duplicated()] 
    for var, (flag, valor) in grupo_a.items():
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
    grupo_b = {
        'HI020': 'HI020_F',  # Motivo aumento → solo si ingresos aumentaron
        'HI030': 'HI030_F',  # Motivo disminución → solo si ingresos disminuyeron
    }

    for var, flag in grupo_b.items():
        if flag in df.columns and var in df.columns:
            mask = df[flag] == -2
            df.loc[mask, var] = np.nan
    #-------------------------------------------------
    # **Grupo C — flag -1 (y -6 en PL060) → NaN
    #-------------------------------------------------
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


    # ══════════════════════════════════════════════════════════════════════════════
    # 6. ELIMINAR FLAGS _F Y COLUMNAS AUXILIARES
    # ══════════════════════════════════════════════════════════════════════════════

    cols_eliminar = [c for c in df.columns if c.endswith('_F')] + ['id_hogar_join']
    df = df.drop(columns=[c for c in cols_eliminar if c in df.columns])


    # ══════════════════════════════════════════════════════════════════════════════
    # 7. RENAME A SNAKE_CASE
    # ══════════════════════════════════════════════════════════════════════════════

    # Variable RENAME_MAP en mapeo_utils.py
    df = df.rename(columns=RENAME_MAP)


    # ══════════════════════════════════════════════════════════════════════════════
    # 8. DECODIFICACIÓN DE VALORES CATEGÓRICOS
    #    Los CSVs del INE vienen con todas las columnas como string.
    #    Variables continuas (rentas, horas, etc.) se convierten a numérico sin decodificar.
    # ══════════════════════════════════════════════════════════════════════════════

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

    for col, mapping in DECODIFICACIONES.items():
        if col in df.columns:
            # Convertir float a int antes de hacer str, para evitar '1.0' en lugar de '1'
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].where(df[col].isna(), df[col].astype('Int64').astype(str))
            else:
                df[col] = df[col].astype(str).where(df[col].notna(), other=np.nan)
            df[col] = df[col].map(mapping)


    # ══════════════════════════════════════════════════════════════════════════════
    # 10. EXPORTACIÓN
    # ══════════════════════════════════════════════════════════════════════════════

    PATH_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PATH_OUT, index=False, encoding='utf-8-sig')

    print(f'\n{'═'*60}')
    print(f'Dataset analítico guardado en: {PATH_OUT}')
    print(f'Filas: {len(df):,} | Columnas: {len(df.columns)}')
    print(f'\nColumnas finales:')
    for col in df.columns:
        dtype = df[col].dtype
        nulls = df[col].isna().sum()
        print(f'  {col:<40} {str(dtype):<10} nulos: {nulls:,}')