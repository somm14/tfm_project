"""
Pipeline de carga y mapeo de microdatos ECV 2025 (INE)
TFM: Predicción de estrés financiero - Comunidad de Madrid
=============================================================
El ZIP del INE contiene varios ficheros CSV:
  - esudb25d.csv  → Fichero D: Datos básicos del hogar
  - esudb25h.csv  → Fichero H: Datos del hogar (renta, vivienda, exclusión)
  - esudb25r.csv  → Fichero R: Datos básicos de la persona
  - esudb25p.csv  → Fichero P: Datos de la persona (trabajo, salud, renta)

Ajusta los nombres de fichero si difieren en tu descarga.
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# 1. CARGA DE FICHEROS
# ─────────────────────────────────────────────

def cargar_ficheros(ruta_base: str) -> dict:
    """
    Carga los cuatro ficheros CSV del ZIP del INE.
    ruta_base: carpeta donde descomprimiste el ZIP, p.ej. "data/ecv2025/"
    """
    ficheros = {
        "D": pd.read_csv(f"{ruta_base}esudb25d.csv", sep=",", low_memory=False),
        "H": pd.read_csv(f"{ruta_base}esudb25h.csv", sep=",", low_memory=False),
        "R": pd.read_csv(f"{ruta_base}esudb25r.csv", sep=",", low_memory=False),
        "P": pd.read_csv(f"{ruta_base}esudb25p.csv", sep=",", low_memory=False),
    }
    for nombre, df in ficheros.items():
        print(f"Fichero {nombre}: {df.shape[0]:,} filas | {df.shape[1]} columnas")
    return ficheros


# ─────────────────────────────────────────────
# 2. FILTRO: COMUNIDAD DE MADRID
# ─────────────────────────────────────────────

def filtrar_madrid(ficheros: dict) -> dict:
    """
    Filtra el fichero D por DB040 == 'ES30' (Comunidad de Madrid)
    y propaga el filtro al resto de ficheros por el ID del hogar.
    """
    # Hogares de Madrid
    d_madrid = ficheros["D"][ficheros["D"]["DB040"] == "ES30"].copy()
    hogares_madrid = set(d_madrid["DB030"])
    print(f"\nHogares en Madrid: {len(hogares_madrid):,}")

    # Filtrar H por hogar
    h_madrid = ficheros["H"][ficheros["H"]["HB030"].isin(hogares_madrid)].copy()

    # El ID de persona en R y P incluye el ID del hogar como prefijo
    # Reconstruimos el ID de hogar desde RB030/PB030 (primeros 6 dígitos)
    r = ficheros["R"].copy()
    r["hogar_id"] = r["RB030"].astype(str).str[:6].astype(int)
    r_madrid = r[r["hogar_id"].isin(hogares_madrid)].copy()

    p = ficheros["P"].copy()
    p["hogar_id"] = p["PB030"].astype(str).str[:6].astype(int)
    p_madrid = p[p["hogar_id"].isin(hogares_madrid)].copy()

    print(f"Personas (fichero R) en Madrid: {len(r_madrid):,}")
    print(f"Personas (fichero P) en Madrid: {len(p_madrid):,}")

    return {"D": d_madrid, "H": h_madrid, "R": r_madrid, "P": p_madrid}


# ─────────────────────────────────────────────
# 3. SELECCIÓN Y RENOMBRADO DE VARIABLES
# ─────────────────────────────────────────────

# Variables del fichero H (hogar) relevantes para estrés financiero
VARS_H = {
    "HB030": "id_hogar",
    "HS120": "dificultad_fin_de_mes",       # Variable objetivo principal (1-6)
    "HS060": "no_puede_gasto_imprevisto",   # 1=Sí puede, 2=No puede
    "HS040": "no_puede_vacaciones",         # 1=Sí puede, 2=No puede
    "HS011": "retraso_hipoteca_alquiler",   # 1=1 vez, 2=2+, 3=No
    "HS021": "retraso_facturas",            # 1=1 vez, 2=2+, 3=No
    "HS031": "retraso_prestamos",           # 1=1 vez, 2=2+, 3=No
    "HS140": "carga_gastos_vivienda",       # 1=Pesada, 2=Razonable, 3=Ninguna
    "HH021": "regimen_tenencia",            # 1=Prop sin hip, 2=Prop con hip, 3=Alquiler mercado...
    "HH060": "importe_alquiler",            # Euros mensuales
    "HH070": "gastos_totales_vivienda",     # Euros mensuales
    "HY020":  "renta_disponible_hogar",     # Renta neta total hogar año anterior
    "HX040": "num_miembros_hogar",
    "HX060": "tipo_hogar",
    "HC050": "capacidad_ahorro",            # 1=Ahorra, 2=Usa ahorros, 3=Pide prestado, 4=Neutro
    "HV080": "meses_sin_ingresos_aguanta",  # 1=<3m, 2=3-5m, 3=6-12m, 4=>12m
    "vhPobreza": "en_riesgo_pobreza",       # 1=Sí, 0=No
    "vhMATDEP": "carencia_material_severa", # 1=Sí, 0=No
}

# Variables del fichero P (persona) relevantes
VARS_P = {
    "PB030": "id_persona",
    "hogar_id": "id_hogar",
    "PB040": "peso_transversal",            # Factor de ponderación — IMPORTANTE para análisis
    "PB140": "anio_nacimiento",
    "PB150": "sexo",                        # 1=Varón, 2=Mujer
    "PB190": "estado_civil",               # 1=Soltero, 2=Casado...
    "PB210": "pais_nacimiento",            # 1=España, 2=UE, 3=Resto
    "PE040": "nivel_estudios",             # 000=Sin estudios ... 500=Superior
    "PL031": "situacion_actividad",        # 1=Asal TC, 2=Asal TP, 3=CxP TC...
    "PL040": "situacion_profesional",      # 1=Empleador, 2=Autónomo, 3=Asalariado
    "PL111A": "sector_actividad",          # NACE: a=Agricultura, c=Industria, g=Comercio...
    "PL140": "tipo_contrato",              # 1=Indefinido, 2=Temporal, -4=Sin contrato
    "PL160": "cambio_trabajo_ultimo_anio", # 1=Sí, 2=No  ← proxy rotación
    "PL170": "motivo_cambio_trabajo",      # 1=Mejor trabajo, 2=Fin contrato, 3=Causas empresa...
    "PL180": "cambio_situacion_actividad", # 1=Empleado→Parado, 4=Parado→Empleado...
    "PL060": "horas_semana",
    "PL073": "meses_asal_tc",             # Meses como asalariado TC en año ref.
    "PL074": "meses_asal_tp",             # Meses como asalariado TP en año ref.
    "PL080": "meses_desempleo",           # Meses en desempleo en año ref.
    "PL130": "tamanio_empresa",           # 1-10=1a10 trabajadores ... 13=50+
    "PL230": "sector_publico_privado",    # 1=Público, 2=Privado
    "PY010N": "salario_neto_anual",       # Renta neta asalariado año anterior
    "PH010": "estado_salud",              # 1=Muy bueno ... 5=Muy malo
    "PH040": "no_consulto_medico",        # 1=Sí (no pudo), 2=No
    "PH050": "motivo_no_medico",          # 1=No se lo podía permitir...
}


def seleccionar_variables(ficheros_madrid: dict) -> tuple:
    """Selecciona y renombra las variables de interés."""

    # Fichero H
    cols_h = [c for c in VARS_H.keys() if c in ficheros_madrid["H"].columns]
    h = ficheros_madrid["H"][cols_h].rename(columns={k: v for k, v in VARS_H.items() if k in cols_h})

    # Fichero P
    cols_p = [c for c in VARS_P.keys() if c in ficheros_madrid["P"].columns]
    p = ficheros_madrid["P"][cols_p].rename(columns={k: v for k, v in VARS_P.items() if k in cols_p})

    print(f"\nVariables hogar seleccionadas: {h.shape[1]}")
    print(f"Variables persona seleccionadas: {p.shape[1]}")

    return h, p


# ─────────────────────────────────────────────
# 4. JOIN HOGAR + PERSONA
# ─────────────────────────────────────────────

def unir_hogar_persona(h: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """Une los datos del hogar con los datos individuales de cada persona."""
    df = p.merge(h, on="id_hogar", how="left")
    print(f"\nDataset unido: {df.shape[0]:,} personas x {df.shape[1]} variables")
    return df


# ─────────────────────────────────────────────
# 5. FILTRO: SOLO PERSONAS OCUPADAS
# ─────────────────────────────────────────────

def filtrar_ocupados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra solo personas asalariadas activas (códigos 1 y 2 en situacion_actividad).
    Son las que tienen sentido para el modelo de estrés financiero laboral.
    """
    if "situacion_actividad" in df.columns:
        df_ocupados = df[df["situacion_actividad"].isin([1, 2])].copy()
        print(f"\nPersonas asalariadas activas (Madrid): {len(df_ocupados):,}")
        return df_ocupados
    else:
        print("AVISO: columna situacion_actividad no encontrada, revisa el mapeo")
        return df


# ─────────────────────────────────────────────
# 6. CONSTRUCCIÓN DE LA VARIABLE OBJETIVO
# ─────────────────────────────────────────────

def construir_variable_objetivo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Variable objetivo: ESTRÉS FINANCIERO ALTO (binaria)

    Definición: una persona tiene estrés financiero alto si cumple
    AL MENOS DOS de estas condiciones:
      - Llega a fin de mes con dificultad o mucha dificultad (HS120 <= 2)
      - No puede afrontar gastos imprevistos (HS060 == 2)
      - Ha tenido retrasos en facturas o alquiler (HS021 o HS011 en [1,2])
      - No puede ahorrar (HC050 en [2,3])

    Usar múltiples indicadores es más robusto que uno solo.
    """
    df = df.copy()

    cond1 = df["dificultad_fin_de_mes"].isin([1, 2])           # Mucha/bastante dificultad
    cond2 = df["no_puede_gasto_imprevisto"] == 2                # No puede afrontar imprevistos
    cond3 = df["retraso_facturas"].isin([1, 2])                 # Retrasos en facturas
    cond4 = df["retraso_hipoteca_alquiler"].isin([1, 2])        # Retrasos en vivienda
    cond5 = df["capacidad_ahorro"].isin([2, 3])                 # Usa ahorros o pide prestado

    # Suma de condiciones cumplidas
    df["n_condiciones_estres"] = (
        cond1.astype(int) + cond2.astype(int) + cond3.astype(int) +
        cond4.astype(int) + cond5.astype(int)
    )

    # Variable objetivo: 1 si cumple 2 o más condiciones
    df["estres_financiero_alto"] = (df["n_condiciones_estres"] >= 2).astype(int)

    tasa = df["estres_financiero_alto"].mean() * 100
    print(f"\nTasa de estrés financiero alto en muestra: {tasa:.1f}%")
    print(df["estres_financiero_alto"].value_counts())

    return df


# ─────────────────────────────────────────────
# 7. LIMPIEZA BÁSICA
# ─────────────────────────────────────────────

def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reemplaza los códigos de valor perdido del INE (-1, -2, -5, -6)
    por NaN para que pandas los trate correctamente.
    """
    df = df.copy()
    MISSING_CODES = [-1, -2, -4, -5, -6]
    df.replace(MISSING_CODES, np.nan, inplace=True)

    # Calcular edad a partir del año de nacimiento
    if "anio_nacimiento" in df.columns:
        df["edad"] = 2024 - df["anio_nacimiento"]  # ingresos ref. año anterior (2024)

    print(f"\nValores nulos por columna (top 10):")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

    return df


# ─────────────────────────────────────────────
# 8. PIPELINE COMPLETO
# ─────────────────────────────────────────────

def pipeline_completo(ruta_base: str, guardar_csv: bool = True) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de carga, filtrado, mapeo y limpieza.

    Uso:
        df = pipeline_completo("data/ecv2025/")
    """
    print("=" * 55)
    print("PIPELINE ECV 2025 — Comunidad de Madrid")
    print("=" * 55)

    ficheros       = cargar_ficheros(ruta_base)
    ficheros_mad   = filtrar_madrid(ficheros)
    h, p           = seleccionar_variables(ficheros_mad)
    df             = unir_hogar_persona(h, p)
    df             = filtrar_ocupados(df)
    df             = limpiar_datos(df)
    df             = construir_variable_objetivo(df)

    if guardar_csv:
        df.to_csv("ecv2025_madrid_procesado.csv", index=False)
        print("\nFichero guardado: ecv2025_madrid_procesado.csv")

    print("\n✓ Pipeline completado")
    print(f"  Dataset final: {df.shape[0]:,} personas x {df.shape[1]} variables")
    print(f"  Variable objetivo: 'estres_financiero_alto'")

    return df


# ─────────────────────────────────────────────
# EJECUCIÓN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Cambia esta ruta por la carpeta donde descomprimiste el ZIP del INE
    RUTA = "data/ecv2025/"
    df = pipeline_completo(RUTA)

    # Vista previa
    print("\nPrimeras filas:")
    print(df[["id_persona", "edad", "sexo", "tipo_contrato",
              "salario_neto_anual", "dificultad_fin_de_mes",
              "estres_financiero_alto"]].head(10))
