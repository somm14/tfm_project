"""
gold_ecv.py
===========
Pipeline end-to-end Gold para la ECV 2025 del INE.

Lee el dataset Silver y produce el dataset listo para modelado.

Uso
---
    python gold_ecv.py                         # rutas por defecto
    python gold_ecv.py --silver ruta/silver.csv --output_dir data/03_gold
    python gold_ecv.py --verbose               # log detallado

Estructura interna
------------------
    1.  Carga del dataset Silver
    2.  Construcción del target `estres_financiero_alto`
    3.  Eliminación de variables no útiles
    4a. Feature engineering — renta hogar per cápita
    4b. Feature engineering — ratio carga de vivienda
    4c. Feature engineering — precariedad laboral
    4d. Feature engineering — agrupación nivel de estudios
    4e. Feature engineering — transformación log1p de rentas
    5.  Tratamiento de nulos residuales
    6a. Encoding ordinal
    6b. Encoding binario
    6c. One-hot encoding
    7.  Exportación

Decisiones documentadas en: notebooks/03_gold_layer.ipynb
"""

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline Gold ECV 2025")
    parser.add_argument(
        "--silver",
        type=Path,
        default=Path("data/02_silver/dataset_analitico.csv"),
        help="Ruta al dataset Silver de entrada",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/03_gold"),
        help="Directorio de salida del dataset Gold",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Imprime log detallado de cada paso",
    )
    return parser.parse_args()


def setup_logging(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )
    return logging.getLogger("gold_ecv")


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────────────────────

# Condiciones de estrés por componente (valores que activan el indicador)
COMPONENTES_ESTRES: dict[str, list[str]] = {
    "capacidad_fin_de_mes":         ["Con mucha dificultad", "Con dificultad"],
    "capacidad_gastos_imprevistos": ["No (no puede)"],
    "retrasos_facturas":            ["Sí, una vez", "Sí, dos o más veces"],
    "retrasos_hipoteca_alquiler":   ["Sí, una vez", "Sí, dos o más veces"],
    "retrasos_deudas_no_vivienda":  ["Sí, una vez", "Sí, dos o más veces"],
}

# Variables a eliminar con justificación
COLS_CONSTANTES:  list[str] = ["region", "situacion_actividad", "situacion_profesional"]
COLS_IDS:         list[str] = ["id_hogar", "id_persona"]
COLS_DUPLICADAS:  list[str] = ["renta_hogar_indicadores"]   # correlación 1.0 con renta_neta_hogar
# Los componentes del target se eliminan en el mismo paso de construcción del target

# Variables continuas a transformar con log1p (skew > 2.5)
COLS_LOG1P: list[str] = [
    "renta_neta_salarial",
    "renta_no_monetaria_salarial",
    "renta_neta_hogar",
    "renta_hogar_per_capita",    # creada en G.4a
    "importe_alquiler",
    "cuota_hipoteca",
    "gastos_vivienda",
]

# Encoding ordinal: variable → {etiqueta: valor numérico}
ENCODING_ORDINAL: dict[str, dict[str, int]] = {
    "nivel_estudios": {
        "Hasta primaria": 0, "Secundaria 1ª etapa": 1, "Post-secundaria": 2,
    },
    "estado_salud": {
        "Muy malo": 0, "Malo": 1, "Regular": 2, "Bueno": 3, "Muy bueno": 4,
    },
    "limitacion_actividad": {
        "Gravemente limitado": 0, "Limitado (no grave)": 1, "No limitado": 2,
    },
    "grado_urbanizacion": {
        "Zona poco poblada": 0, "Zona media": 1, "Zona muy poblada": 2,
    },
    "cambio_ingresos_12m": {
        "Han disminuido": 0, "Se mantienen": 1, "Han aumentado": 2,
    },
    "expectativa_ingresos_12m": {
        "Empeorar": 0, "Mantenerse": 1, "Mejorar": 2,
    },
    "carga_prestamos_no_vivienda": {
        "Una carga pesada": 0, "Una carga razonable": 1, "Ninguna carga": 2,
    },
    "carga_asistencia_medica": {
        "Una carga pesada": 0, "Una carga razonable": 1,
        "Ninguna carga": 2,    "No ha utilizado": 3,
    },
    "carga_asistencia_dental": {
        "Una carga pesada": 0, "Una carga razonable": 1,
        "Ninguna carga": 2,    "No ha utilizado": 3,
    },
    "carga_medicamentos": {
        "Una carga pesada": 0, "Una carga razonable": 1,
        "Ninguna carga": 2,    "No ha consumido": 3,
    },
}

# Encoding binario: mapa global de valores a 0/1
MAPA_BINARIO: dict[str, float] = {
    "Sí": 1, "No": 0,
    "Hombre": 1, "Mujer": 0,
    "Tiempo completo": 1, "Tiempo parcial": 0,
    "No aplicable (≥60 años)": np.nan,
}

# Variables que reciben encoding binario
COLS_BINARIAS: list[str] = [
    "sexo", "jornada", "personal_a_cargo", "enfermedad_cronica",
    "necesito_medico_no_fue", "puede_vacaciones", "puede_proteina_2dias",
    "puede_calefaccion_invierno", "hogar_riesgo_pobreza", "hogar_carencia_material",
    "arope_2020", "arope_2030", "carencia_material_social_severa",
    "baja_intensidad_laboral_2020",
]

# Agrupación de nivel de estudios
MAPA_ESTUDIOS: dict[str, str] = {
    "Sin estudios":                 "Hasta primaria",
    "Primaria incompleta":          "Hasta primaria",
    "Primaria":                     "Hasta primaria",
    "Secundaria 1ª etapa":          "Secundaria 1ª etapa",
    "Secundaria 1ª etapa (título)": "Secundaria 1ª etapa",
    "Secundaria 2ª etapa (gral)":   "Post-secundaria",
    "Post-secundaria no superior":  "Post-secundaria",
}


# ──────────────────────────────────────────────────────────────────────────────
# PASO 1 — CARGA
# ──────────────────────────────────────────────────────────────────────────────

def cargar_silver(path: Path, log: logging.Logger) -> pd.DataFrame:
    log.info("PASO 1 — Carga del dataset Silver")
    df = pd.read_csv(path, low_memory=False)
    log.info(f"  {df.shape[0]:,} filas × {df.shape[1]} columnas")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# PASO 2 — CONSTRUCCIÓN DEL TARGET
# ──────────────────────────────────────────────────────────────────────────────

def construir_target(df: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    """
    Construye `estres_financiero_alto` (binaria 0/1).

    Regla: ≥ 2 de 5 condiciones activas simultáneamente.
    Si con los componentes disponibles el score ya es ≥ 2 → 1.
    Si los componentes disponibles son ≥ 4 y el score es < 2 → 0.
    En cualquier otro caso → NaN (indeterminable).

    Los 5 componentes se eliminan del dataframe tras la construcción
    para evitar data leakage en el modelo.
    """
    log.info("PASO 2 — Construcción del target")

    comp_cols = []
    for col, vals in COMPONENTES_ESTRES.items():
        nombre = f"_comp_{col}"
        df[nombre] = df[col].isin(vals)
        comp_cols.append(nombre)

    score              = df[comp_cols].sum(axis=1, skipna=True)
    n_disponibles      = df[comp_cols].notna().sum(axis=1)

    df["estres_financiero_alto"] = np.where(
        score >= 2,
        1,
        np.where((score < 2) & (n_disponibles >= 4), 0, np.nan),
    )

    # Limpiar auxiliares y componentes (leakage)
    df = df.drop(columns=comp_cols + list(COMPONENTES_ESTRES.keys()))

    n0  = int((df["estres_financiero_alto"] == 0).sum())
    n1  = int((df["estres_financiero_alto"] == 1).sum())
    nan = int(df["estres_financiero_alto"].isna().sum())
    log.info(f"  0 (sin estrés): {n0:,}  ({n0/len(df)*100:.1f}%)")
    log.info(f"  1 (estrés):     {n1:,}  ({n1/len(df)*100:.1f}%)")
    log.info(f"  NaN:            {nan:,}  — ratio desbalanceo 1:{n0/n1:.1f}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# PASO 3 — ELIMINACIÓN DE VARIABLES
# ──────────────────────────────────────────────────────────────────────────────

def eliminar_variables(df: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    """
    Elimina columnas no útiles para el modelo:
      - Constantes (cero varianza tras los filtros Silver)
      - Identificadores
      - Variable duplicada (renta_hogar_indicadores ≡ renta_neta_hogar)
    """
    log.info("PASO 3 — Eliminación de variables no útiles")
    eliminar = COLS_CONSTANTES + COLS_IDS + COLS_DUPLICADAS
    presentes = [c for c in eliminar if c in df.columns]
    df = df.drop(columns=presentes)
    log.info(f"  Eliminadas: {presentes}")
    log.info(f"  Columnas restantes: {df.shape[1]}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# PASO 4 — FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────

def feature_engineering(df: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    log.info("PASO 4 — Feature engineering")

    # 4a. Renta hogar per cápita (escala OCDE modificada)
    df["renta_hogar_per_capita"] = df["renta_neta_hogar"] / df["unidades_consumo"]
    log.debug("  [4a] renta_hogar_per_capita creada")

    # 4b. Ratio carga de vivienda (anualizado sobre renta salarial neta)
    # Capeado al P99 para evitar distorsión por rentas muy bajas
    df["ratio_carga_vivienda"] = np.where(
        df["renta_neta_salarial"] > 0,
        (df["gastos_vivienda"] * 12) / df["renta_neta_salarial"],
        np.nan,
    )
    p99 = df["ratio_carga_vivienda"].quantile(0.99)
    df["ratio_carga_vivienda"] = df["ratio_carga_vivienda"].clip(upper=p99)
    n_sobrecarga = (df["ratio_carga_vivienda"] > 0.30).sum()
    log.debug(f"  [4b] ratio_carga_vivienda creada  — {n_sobrecarga} personas >30% (sobrecarga)")

    # 4c. Indicador de precariedad laboral (contrato temporal O jornada parcial)
    es_temporal = df["tipo_contrato"].isin(["Temporal escrito", "Temporal verbal"])
    es_parcial  = df["jornada"] == "Tiempo parcial"
    df["precariedad_laboral"] = (es_temporal | es_parcial)
    log.debug(f"  [4c] precariedad_laboral — {df['precariedad_laboral'].sum()} personas")

    # 4d. Agrupación de nivel_estudios (eliminar categorías con n < 20)
    df["nivel_estudios"] = df["nivel_estudios"].map(MAPA_ESTUDIOS)
    log.debug("  [4d] nivel_estudios agrupado en 3 categorías")

    # 4e. Transformación log1p sobre rentas e importes con skew alto
    for col in COLS_LOG1P:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))
    log.info(f"  Features nuevas: renta_hogar_per_capita, ratio_carga_vivienda, "
             f"precariedad_laboral, log_* ({sum(1 for c in COLS_LOG1P if c in df.columns)} vars)")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# PASO 5 — TRATAMIENTO DE NULOS RESIDUALES
# ──────────────────────────────────────────────────────────────────────────────

def tratar_nulos(df: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    """
    Dos grupos de tratamiento:

    Grupo II (nulos estructurales persistentes):
      - motivo_aumento_ingresos / motivo_disminucion_ingresos:
        NaN = "no hubo cambio" → categoría explícita 'No aplica'
      - expectativa_ingresos_12m (módulo opcional, 9.8% NaN):
        → indicador binario de no-respuesta + imputación con moda

    Grupo I (nulos informativos, < 2%):
      - Numéricas → mediana
      - Categóricas → moda
      IMPORTANTE: en producción, ajustar mediana/moda solo sobre train set
    """
    log.info("PASO 5 — Tratamiento de nulos residuales")

    # Grupo II
    df["motivo_aumento_ingresos"]     = df["motivo_aumento_ingresos"].fillna("No aplica (sin aumento)")
    df["motivo_disminucion_ingresos"] = df["motivo_disminucion_ingresos"].fillna("No aplica (sin disminución)")

    df["expectativa_sin_respuesta"] = df["expectativa_ingresos_12m"].isna().astype(int)
    moda_exp = df["expectativa_ingresos_12m"].mode()[0]
    df["expectativa_ingresos_12m"]  = df["expectativa_ingresos_12m"].fillna(moda_exp)

    log.debug("  [II] motivos ingresos → categoría 'No aplica'")
    log.debug(f"  [II] expectativa_sin_respuesta creada; moda={repr(moda_exp)}")

    # Grupo I: nulos informativos
    nulos = df.isnull().sum()
    nulos = nulos[(nulos > 0) & (nulos.index != "estres_financiero_alto")]

    cols_num = df.select_dtypes(include="number").columns
    cols_cat = df.select_dtypes(include="object").columns
    n_imputadas = 0

    for col in nulos.index:
        n = int(df[col].isna().sum())
        if col in cols_num:
            val = df[col].median()
            df[col] = df[col].fillna(val)
        elif col in cols_cat:
            val = df[col].mode()[0]
            df[col] = df[col].fillna(val)
        else:
            continue
        n_imputadas += n
        log.debug(f"  [I]  {col}: {n} nulos → {repr(val)}")

    log.info(f"  Grupo I: {n_imputadas} imputaciones (mediana/moda)")
    nulos_restantes = df.drop(columns=["estres_financiero_alto"]).isnull().sum().sum()
    log.info(f"  Nulos restantes en features: {nulos_restantes}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# PASO 6 — ENCODING
# ──────────────────────────────────────────────────────────────────────────────

def encodificar(df: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    log.info("PASO 6 — Encoding de variables categóricas")

    # 6a. Ordinal
    n_ord = 0
    for col, mapa in ENCODING_ORDINAL.items():
        if col in df.columns:
            df[col] = df[col].map(mapa)
            n_ord += 1
    log.info(f"  [6a] Ordinal: {n_ord} variables")

    # 6b. Binario
    n_bin = 0
    for col in COLS_BINARIAS:
        if col in df.columns:
            df[col] = df[col].map(MAPA_BINARIO)
            n_bin += 1
    log.info(f"  [6b] Binario: {n_bin} variables")

    # Eliminar anio_nacimiento (redundante con edad)
    if "anio_nacimiento" in df.columns:
        df = df.drop(columns=["anio_nacimiento"])
        log.debug("  anio_nacimiento eliminado (redundante con edad)")

    # 6c. One-hot: columnas object restantes (excluir target y peso)
    ya_encodadas = set(ENCODING_ORDINAL.keys()) | set(COLS_BINARIAS)
    excluir      = ya_encodadas | {"estres_financiero_alto", "peso_persona"}
    cols_ohe     = [c for c in df.select_dtypes(include="object").columns
                    if c not in excluir]

    n_cols_pre = df.shape[1]
    df = pd.get_dummies(df, columns=cols_ohe, drop_first=False, dtype=int)
    n_cols_ohe = df.shape[1] - n_cols_pre
    log.info(f"  [6c] One-hot: {len(cols_ohe)} variables → +{n_cols_ohe} columnas dummy")
    log.info(f"  Total columnas tras encoding: {df.shape[1]}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# PASO 7 — EXPORTACIÓN
# ──────────────────────────────────────────────────────────────────────────────

def imputar_post_encoding(df: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    """
    Imputa NaN generados por el encoding binario.

    Caso concreto: baja_intensidad_laboral_2020 tiene la categoría
    'No aplicable (>=60 anos)' que el MAPA_BINARIO convierte a NaN correctamente,
    pero que el paso 5 no puede tratar porque en ese momento la columna es aun string.
    Se imputa con la mediana (0 = No).
    """
    log.info("PASO 6d — Imputacion post-encoding")
    cols_num  = df.select_dtypes(include="number").columns
    nulos_enc = {c: int(df[c].isna().sum())
                 for c in cols_num
                 if c != "estres_financiero_alto" and df[c].isna().sum() > 0}

    n_total = 0
    for col, n in nulos_enc.items():
        val = df[col].median()
        df[col] = df[col].fillna(val)
        n_total += n
        log.debug(f"  {col}: {n} NaN post-encoding → mediana={val}")

    if n_total:
        log.info(f"  {n_total} NaN post-encoding imputados con mediana")
    else:
        log.info("  Sin NaN post-encoding")
    return df


def exportar(df: pd.DataFrame, output_dir: Path, log: logging.Logger) -> Path:
    log.info("PASO 7 — Exportación")
    output_dir.mkdir(parents=True, exist_ok=True)
    path_out = output_dir / "dataset_modelado.csv"
    df.to_csv(path_out, index=False, encoding="utf-8-sig")
    log.info(f"  ✓  {path_out}")
    log.info(f"     {df.shape[0]:,} filas × {df.shape[1]} columnas")
    return path_out


# ──────────────────────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ──────────────────────────────────────────────────────────────────────────────

def imprimir_resumen(df: pd.DataFrame, log: logging.Logger) -> None:
    y    = df["estres_financiero_alto"]
    X    = df.drop(columns=["estres_financiero_alto", "peso_persona"], errors="ignore")
    n0   = int((y == 0).sum())
    n1   = int((y == 1).sum())
    nnan = int(y.isna().sum())

    log.info("─" * 60)
    log.info("RESUMEN DEL DATASET GOLD")
    log.info(f"  Observaciones:       {len(df):,}")
    log.info(f"  Features (X):        {X.shape[1]}")
    log.info(f"  Target (y):")
    log.info(f"    0 — sin estrés:    {n0:,}  ({n0/len(df)*100:.1f}%)")
    log.info(f"    1 — estrés alto:   {n1:,}  ({n1/len(df)*100:.1f}%)")
    log.info(f"    NaN:               {nnan:,}")
    log.info(f"  Ratio desbalanceo:   1:{n0/n1:.1f}")
    log.info(f"  Nulos en X:          {X.isnull().sum().sum():,}")
    log.info("─" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    log  = setup_logging(args.verbose)

    log.info("=" * 60)
    log.info("PIPELINE GOLD — ECV 2025 (INE)")
    log.info("=" * 60)

    df = cargar_silver(args.silver, log)
    df = construir_target(df, log)
    df = eliminar_variables(df, log)
    df = feature_engineering(df, log)
    df = tratar_nulos(df, log)
    df = encodificar(df, log)
    df = imputar_post_encoding(df, log)
    exportar(df, args.output_dir, log)
    imprimir_resumen(df, log)


if __name__ == "__main__":
    main()
