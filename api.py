"""
api.py
======
API FastAPI para el modelo de predicción de estrés financiero (TFM ECV 2025).

Endpoints
─────────
  GET  /              → health check
  POST /predict       → predicción individual (JSON con features crudas)
  POST /predict/batch → predicción en batch (lista de registros)

Uso local
─────────
  uvicorn api:app --reload --port 8000

Despliegue en Render
────────────────────
  Start command: uvicorn api:app --host 0.0.0.0 --port $PORT
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CARGA DEL MODELO (una sola vez al arrancar)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = Path(os.getenv("MODEL_PATH", "src/models/lighgbm_grid.pkl"))

try:
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)
    logger.info(f"Modelo cargado desde {MODEL_PATH}")
except FileNotFoundError:
    logger.error(f"No se encontró el modelo en {MODEL_PATH}")
    pipeline = None

# ─────────────────────────────────────────────────────────────────────────────
# UMBRALES Y ETIQUETAS
# ─────────────────────────────────────────────────────────────────────────────
UMBRAL_DEFAULT = float(os.getenv("UMBRAL", "0.5"))

def score_a_nivel(score: float) -> str:
    if score >= 0.70:
        return "Alto"
    elif score >= 0.40:
        return "Medio"
    else:
        return "Bajo"

# ─────────────────────────────────────────────────────────────────────────────
# COLUMNAS QUE EL MODELO EXCLUYE (mismas que en modelo_produccion.py)
# ─────────────────────────────────────────────────────────────────────────────
FEATURES_DEL = [
    'motivo_aumento_ingresos', 'motivo_disminucion_ingresos',
    'id_hogar', 'id_persona', 'region',
    'capacidad_fin_de_mes', 'capacidad_gastos_imprevistos',
    'retrasos_facturas', 'retrasos_hipoteca_alquiler',
    'retrasos_deudas_no_vivienda',
    'estres_financiero_alto', 'peso_persona',
]

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA DE ENTRADA
# Todos los campos son Optional para máxima flexibilidad:
# el pipeline sklearn ya tiene SimpleImputer para gestionar nulos.
# ─────────────────────────────────────────────────────────────────────────────
class InputFeatures(BaseModel):
    # Demográficas
    edad:                       Optional[float] = None
    sexo:                       Optional[str]   = None   # 'Hombre' / 'Mujer'
    pais_nacimiento:            Optional[str]   = None
    nacionalidad:               Optional[str]   = None

    # Situación laboral
    horas_semana:               Optional[float] = None
    jornada:                    Optional[str]   = None   # 'Tiempo completo' / 'Tiempo parcial'
    tipo_contrato:              Optional[str]   = None   # 'Temporal escrito', 'Indefinido escrito', etc.
    personal_a_cargo:           Optional[str]   = None   # 'Sí' / 'No'
    anios_experiencia:          Optional[float] = None
    meses_desempleo_ref:        Optional[float] = None
    meses_desempleo_5anios:     Optional[float] = None
    ocupacion_isco08:           Optional[float] = None
    sector_cnae:                Optional[str]   = None

    # Educación
    nivel_estudios:             Optional[str]   = None   # 'Hasta primaria', 'Superior universitario', etc.

    # Salud
    estado_salud:               Optional[str]   = None   # 'Muy bueno' … 'Muy malo'
    enfermedad_cronica:         Optional[str]   = None   # 'Sí' / 'No'
    limitacion_actividad:       Optional[str]   = None
    necesito_medico_no_fue:     Optional[str]   = None   # 'Sí' / 'No'

    # Renta individual
    renta_neta_salarial:        Optional[float] = None
    renta_no_monetaria_salarial: Optional[float] = None

    # Renta hogar
    renta_neta_hogar:           Optional[float] = None
    renta_hogar_indicadores:    Optional[float] = None
    unidades_consumo:           Optional[float] = None
    num_miembros_hogar:         Optional[float] = None
    tipo_hogar:                 Optional[str]   = None

    # Indicadores pobreza (INE derivados — entran como feature, no como target)
    hogar_riesgo_pobreza:       Optional[str]   = None   # 'Sí' / 'No'
    hogar_carencia_material:    Optional[str]   = None

    # Privación material
    puede_vacaciones:           Optional[str]   = None   # 'Sí' / 'No'
    puede_proteina_2dias:       Optional[str]   = None
    tiene_ordenador:            Optional[str]   = None
    tiene_coche:                Optional[str]   = None
    carga_prestamos_no_vivienda: Optional[str]  = None
    puede_sustituir_muebles:    Optional[str]   = None
    puede_calefaccion_invierno: Optional[str]   = None

    # Vivienda
    tipo_vivienda:              Optional[str]   = None
    regimen_tenencia:           Optional[str]   = None
    num_habitaciones:           Optional[float] = None
    importe_alquiler:           Optional[float] = None
    gastos_vivienda:            Optional[float] = None
    cuota_hipoteca:             Optional[float] = None

    # Dinámica ingresos
    cambio_ingresos_12m:        Optional[str]   = None
    expectativa_ingresos_12m:   Optional[str]   = None

    # Carga sanitaria
    carga_asistencia_medica:    Optional[str]   = None
    carga_asistencia_dental:    Optional[str]   = None
    carga_medicamentos:         Optional[str]   = None

    # Variables derivadas persona
    baja_intensidad_laboral_2020: Optional[str] = None
    arope_2020:                 Optional[str]   = None
    carencia_material_social_severa: Optional[str] = None
    arope_2030:                 Optional[str]   = None

    # Features de ingeniería construidas en bronze_to_silver
    renta_hogar_per_capita:     Optional[float] = None
    ratio_carga_vivienda:       Optional[float] = None
    precariedad_laboral:        Optional[float] = None

    # Urbanización
    grado_urbanizacion:         Optional[str]   = None

    model_config = {"extra": "allow"}  # acepta campos adicionales sin romper


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA DE SALIDA
# ─────────────────────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    score_riesgo: float
    clase:        int
    nivel_riesgo: str


class BatchPredictionResponse(BaseModel):
    predicciones: list[PredictionResponse]
    n:            int


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="API Estrés Financiero — TFM ECV 2025",
    description="Predicción de riesgo de estrés financiero alto en asalariados de la Comunidad de Madrid.",
    version="1.0.0",
)

# CORS — necesario para que Lovable pueda llamar a la API desde el navegador
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # En producción, reemplazar por el dominio de Lovable
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def input_to_dataframe(features: InputFeatures) -> pd.DataFrame:
    """Convierte el schema Pydantic en un DataFrame con las columnas que
    espera el pipeline (excluye las columnas que se eliminan en producción)."""
    data = features.model_dump()
    # Eliminar campos que el pipeline nunca vio durante el entrenamiento
    for col in FEATURES_DEL:
        data.pop(col, None)
    return pd.DataFrame([data])


def predecir(df: pd.DataFrame, umbral: float) -> list[PredictionResponse]:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")
    try:
        scores = pipeline.predict_proba(df)[:, 1]
        clases = (scores >= umbral).astype(int)
        return [
            PredictionResponse(
                score_riesgo=round(float(s), 4),
                clase=int(c),
                nivel_riesgo=score_a_nivel(float(s)),
            )
            for s, c in zip(scores, clases)
        ]
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "modelo_cargado": pipeline is not None,
        "umbral_default": UMBRAL_DEFAULT,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Predicción"])
def predict(features: InputFeatures, umbral: float = UMBRAL_DEFAULT):
    """
    Predicción individual.

    - **features**: JSON con las variables del individuo (valores crudos, igual
      que en el dataset silver — el pipeline hace todo el preprocesado internamente).
    - **umbral**: umbral de clasificación (por defecto 0.5, configurable por query param).

    Devuelve `score_riesgo` (0-1), `clase` (0/1) y `nivel_riesgo` (Bajo/Medio/Alto).
    """
    df = input_to_dataframe(features)
    return predecir(df, umbral)[0]


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predicción"])
def predict_batch(registros: list[InputFeatures], umbral: float = UMBRAL_DEFAULT):
    """
    Predicción en batch.

    Acepta una lista de hasta 1.000 registros y devuelve una predicción por cada uno.
    Útil para integraciones de RRHH que quieran evaluar un colectivo completo.
    """
    if len(registros) > 1000:
        raise HTTPException(status_code=400, detail="Máximo 1.000 registros por petición.")
    df = pd.concat([input_to_dataframe(r) for r in registros], ignore_index=True)
    predicciones = predecir(df, umbral)
    return BatchPredictionResponse(predicciones=predicciones, n=len(predicciones))
