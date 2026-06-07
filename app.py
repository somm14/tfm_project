'''
app.py - Demo TFM: Predicción de Estrés Financiero
---------------------------------------------------
Interfaz con login por rol: Empleado / RRHH
    streamlit run app.py
'''

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import uuid
import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GENERAL
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='FinWellsor | Predicción de Estrés Financiero',
    page_icon='💼',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ─────────────────────────────────────────────────────────────────────────────
# ESTILOS CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: "DM Sans", sans-serif; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1923 0%, #1a2d3d 100%);
    border-right: 1px solid #2a3f52;
}
[data-testid="stSidebar"] * { color: #e8edf2 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.9rem; padding: 6px 0; letter-spacing: 0.02em; }

[data-testid="InputInstructions"]  { display:none; }

            
h1 { font-family: "DM Serif Display", serif !important; color: #506478 !important; font-size: 2.4rem !important; letter-spacing: -0.01em; }
h2 { font-family: "DM Serif Display", serif !important; color: #5898db !important; font-size: 1.7rem !important; }
h3 { font-family: "DM Serif Display", sans-serif !important; font-weight: 600 !important; color: #5898db !important; }

.metric-card { background: white; border-radius: 12px; padding: 24px 20px; border-left: 4px solid #0ea5e9; box-shadow: 0 2px 12px rgba(0,0,0,0.06); margin-bottom: 8px; }
.metric-card.danger { border-left-color: #ef4444; }
.metric-card.warning { border-left-color: #f59e0b; }
.metric-card.success { border-left-color: #10b981; }
.metric-card.info { border-left-color: #0ea5e9; }
.metric-value { font-size: 2.2rem; font-weight: 600; color: #0f1923; line-height: 1.1; }
.metric-label { font-size: 0.82rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }

.hero-banner { background: linear-gradient(135deg, #0f1923 0%, #1a4060 60%, #0ea5e9 100%); border-radius: 16px; padding: 48px 40px; margin-bottom: 32px; color: white; }
.hero-banner h1 { color: white !important; font-size: 2.8rem !important; margin-bottom: 8px; }
.hero-banner p { color: #b0d4e8; font-size: 1.1rem; max-width: 600px; }

.badge { display: inline-block; padding: 4px 14px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; }
.badge-alto { background: #fee2e2; color: #b91c1c; }
.badge-medio { background: #fef3c7; color: #92400e; }
.badge-bajo { background: #d1fae5; color: #065f46; }

.prediction-result { border-radius: 16px; padding: 32px; text-align: center; margin: 24px 0; }
.pred-alto { background: linear-gradient(135deg, #fef2f2, #fee2e2); border: 2px solid #fca5a5; }
.pred-medio { background: linear-gradient(135deg, #fffbeb, #fef3c7); border: 2px solid #fcd34d; }
.pred-bajo { background: linear-gradient(135deg, #f0fdf4, #d1fae5); border: 2px solid #6ee7b7; }

.stDataFrame { border-radius: 10px; overflow: hidden; }

.section-divider { height: 3px; background: linear-gradient(90deg, #0ea5e9, transparent); border-radius: 2px; margin: 24px 0 32px 0; }

.info-box { background: #1d3240; border: 1px solid #bae6fd; border-radius: 10px; padding: 18px 20px; margin: 16px 0; }
.warn-box { background: #fffbeb; border: 1px solid #fde68a; border-radius: 10px; padding: 18px 20px; margin: 16px 0; }
.privacy-box { background: #f0fdf4; border: 2px solid #86efac; border-radius: 12px; padding: 24px; margin: 20px 0; }
.consent-box { background: #fafafa; border: 1px solid #d1d5db; border-radius: 10px; padding: 20px; margin: 12px 0; font-size: 0.88rem; color: #374151; line-height: 1.7; }

.step-num { display: inline-flex; align-items: center; justify-content: center; width: 32px; height: 32px; background: #0ea5e9; color: white; border-radius: 50%; font-weight: 700; font-size: 0.9rem; margin-right: 10px; }

.budget-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.budget-table th { background: #0f1923; color: white; padding: 12px 16px; text-align: left; font-weight: 500; letter-spacing: 0.04em; }
.budget-table td { padding: 11px 16px; border-bottom: 1px solid #e5e7eb; }
.budget-table tr:nth-child(even) td { background: #f8fafc; color: #282a2e}
.budget-table tr:last-child td { font-weight: 700; background: #f0f9ff; border-top: 2px solid #0ea5e9; }

.sidebar-logo { font-family: "DM Serif Display", serif; font-size: 1.6rem; color: white; margin-bottom: 4px; }
.sidebar-tagline { font-size: 0.72rem; color: #7faec2; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 28px; }

/* Login card */
.login-card { background: white; border-radius: 20px; padding: 48px 40px; box-shadow: 0 8px 40px rgba(0,0,0,0.12); max-width: 460px; margin: 0 auto; }
.login-role-btn { display: flex; align-items: center; gap: 16px; background: #f8fafc; border: 2px solid #e2e8f0; border-radius: 14px; padding: 20px 24px; margin: 10px 0; cursor: pointer; transition: all 0.2s; }
.login-role-btn:hover { border-color: #0ea5e9; background: #f0f9ff; }
.role-icon { font-size: 2rem; }
.role-title { font-weight: 600; font-size: 1rem; color: #0f172a; }
.role-desc { font-size: 0.82rem; color: #64748b; }

/* Firma digital */
.firma-box { background: #f8fafc; border: 2px dashed #cbd5e1; border-radius: 12px; padding: 24px; text-align: center; margin: 16px 0; }
</style>
''', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CARGA DEL MODELO
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    model_path = Path('src/models/lg_optimo.pkl')
    if model_path.exists():
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

modelo = cargar_modelo()

# ─────────────────────────────────────────────────────────────────────────────
# log1p_rentas — pickle busca esta función por nombre cuando carga el pkl.
# Índices hardcoded como literal local (sin variable global):
#   renta_neta_salarial=2, renta_no_monetaria=13, renta_neta_hogar=6,
#   renta_hogar_per_capita=10, importe_alquiler=4, cuota_hipoteca=3,
#   gastos_vivienda=5
# Tras regenerar el pkl con el nuevo model_pipeline.py (closure), esta
# función ya no será invocada desde el pkl y puede eliminarse.
# ─────────────────────────────────────────────────────────────────────────────
def log1p_rentas(X):
    _idx = [2, 13, 6, 10, 4, 3, 5]  # índices literales — sin globals
    X = X.copy().astype(float)
    X[:, _idx] = np.log1p(np.clip(X[:, _idx], 0, None))
    return X

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS PREDICCIÓN
# ─────────────────────────────────────────────────────────────────────────────
FEATURES_DEL = [
    'motivo_aumento_ingresos', 'motivo_disminucion_ingresos',
    'id_hogar', 'id_persona', 'region',
    'capacidad_fin_de_mes', 'capacidad_gastos_imprevistos',
    'retrasos_facturas', 'retrasos_hipoteca_alquiler',
    'retrasos_deudas_no_vivienda', 'estres_financiero_alto', 'peso_persona',
]

def nivel_riesgo(score: float):
    if score >= 0.70:
        return 'Alto', 'alto'
    elif score >= 0.40:
        return 'Medio', 'medio'
    else:
        return 'Bajo', 'bajo'

def predecir_individuo(data: dict):
    if modelo is None:
        score = round(np.random.beta(2, 3), 4)
        label, css = nivel_riesgo(score)
        return score, label, css
    df = pd.DataFrame([data])
    for col in FEATURES_DEL:
        df.drop(columns=[col], errors='ignore', inplace=True)
    score = float(modelo.predict_proba(df)[:, 1][0])
    label, css = nivel_riesgo(score)
    return score, label, css

def predecir_batch(df: pd.DataFrame) -> pd.DataFrame:
    if modelo is None:
        df['score_riesgo'] = np.nan
        df['nivel_riesgo'] = 'Sin modelo'
        return df
    df_model = df.copy()
    for col in FEATURES_DEL:
        df_model.drop(columns=[col], errors='ignore', inplace=True)
    scores = modelo.predict_proba(df_model)[:, 1]
    df['score_riesgo'] = np.round(scores, 4)
    df['nivel_riesgo'] = [nivel_riesgo(s)[0] for s in scores]
    return df

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES FORMULARIO
# ─────────────────────────────────────────────────────────────────────────────
def composicion_tipo_hogar(estado_civil, num_hijos, num_miembros, edad, sexo):
    if num_miembros == 4 and num_hijos == 2:
        return "2 adultos, 2 niños"
    elif num_miembros == 3 and num_hijos == 1:
        return "2 adultos, 1 niño"
    elif num_miembros == 2 and num_hijos == 0 and edad <= 65:
        return "2 adultos <65 sin niños"
    elif sexo == "Hombre" and 30 <= edad <= 64:
        return "Una persona: hombre 30-64 años"
    elif sexo == "Mujer" and 30 <= edad <= 64:
        return "Una persona: mujer 30-64 años"
    elif num_hijos >= 3 and num_miembros >= 3:
        return "2 adultos, ≥3 niños"
    elif estado_civil in ['Divorciado/a o separado/a', 'Soltero/a', "Viudo/a"] and num_hijos >= 1:
        return "1 adulto con niños"
    elif num_hijos == 0 and num_miembros == 1 and sexo == 'Hombre' and edad <= 30:
        return "Una persona: hombre <30 años"
    elif num_hijos == 0 and num_miembros == 1 and sexo == 'Mujer' and edad <= 30:
        return "Una persona: mujer <30 años"
    else:
        return "Otro tipo de hogar"

def composicion_limitaciones_material(privacion_material):
    resultado = {
        'puede_proteina_2dias': 'Sí',
        'puede_vacaciones': 'Sí',
        'puede_sustituir_muebles': 'Sí',
        'puede_calefaccion_invierno': 'Sí',
        'hogar_carencia_material': 'No'
    }
    if not privacion_material or "Ninguna de las anteriores" in privacion_material:
        return resultado
    seleccion = set(privacion_material)
    if "Comida" in seleccion:
        resultado['puede_proteina_2dias'] = 'No'
        resultado['hogar_carencia_material'] = 'Sí'
    if "Vacaciones" in seleccion:
        resultado['puede_vacaciones'] = 'No'
        resultado['hogar_carencia_material'] = 'Sí'
    if "Reforma hogar" in seleccion:
        resultado['puede_sustituir_muebles'] = 'No (no puede permitírselo)'
        resultado['hogar_carencia_material'] = 'Sí'
    if "Calefacción" in seleccion:
        resultado['puede_calefaccion_invierno'] = 'No'
        resultado['hogar_carencia_material'] = 'Sí'
    return resultado

MAPEO_ESTUDIOS = {
    "Sin estudios": "Sin estudios",
    "Educación primaria": "Hasta primaria",
    "Educación secundaria": "Secundaria 1a etapa",
    "Bachillerato o FP": "post-secundaria",
    "Universidad o superior": "post-secundaria",
}

MAPEO_CONTRATO = {
    "Indefinido": "Indefinido escrito",
    "Temporal": "Temporal escrito",
}

MAPEO_VIVIENDA = {
    "Propietario sin hipoteca": "Propiedad sin hipoteca",
    "Propietario con hipoteca": "Propiedad con hipoteca",
    "Arrendatario": "Alquiler precio mercado",
    "Cesión gratuita": "Cesión gratuita",
    "Alquiler social": "Alquiler precio reducido",
}

MAPEO_DEUDAS = {
    "No": "Ninguna carga",
    "Sí, son manejables": "Una carga razonable",
    "Sí, me generan dificultades": "Una carga pesada",
}

MAPEO_DENTAL = {
    "No tengo gastos de dentista": "Ninguna carga",
    "Los afronto sin problemas": "Una carga razonable",
    "Son una carga importante": "Una carga pesada",
}

MAPEO_MEDICAMENTOS = {
    "No tengo gastos de medicamentos": "Ninguna carga",
    "Los afronto sin problemas": "Una carga razonable",
    "Son una carga importante": "Una carga pesada",
}

MAPEO_CAMBIO_INGRESOS = {
    "Han aumentado": "Aumentado",
    "Se han mantenido igual": "Se mantienen",
    "Han disminuido": "Disminuido",
}

MAPEO_SECTOR = {
    "Dirección y Gerencia": 11,
    "Profesional cualificado (Ingeniería, tecnología, finanzas, leyes, etc.)": 25,
    "Técnico o perfil intermedio (Soporte, diseño, gestión media)": 31,
    "Administración y oficina (Contabilidad, atención al cliente, secretariado)": 41,
    "Ventas y Servicios (Comercial, atención en tienda, hostelería)": 51,
    "Oficios manuales cualificados (Mecánica, mantenimiento, construcción)": 72,
    "Operador de maquinaria o transporte (Conductor, reparto, operario)": 83,
    "Tareas de apoyo o elementales (Limpieza, almacén, auxiliares)": 91,
    "Sector agrícola o pesquero": 61,
}

# ─────────────────────────────────────────────────────────────────────────────
# SIMULACIÓN BBDD CORPORATIVA — salario neto anual
# Basado en medianas reales de la ECV 2025 (salarios_ocupacion.csv),
# segmentadas por grupo ISCO-08 (2 dígitos), jornada y tipo de contrato.
# Genera un valor determinista (reproducible para el mismo ID).
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def _cargar_medianas_salario() -> pd.DataFrame:
    """
    Lee salarios_ocupacion.csv y devuelve la mediana de renta_neta_salarial
    agrupada por (grupo_isco, jornada, contrato_tipo).
    grupo_isco: primer dígito del código ISCO-08 × 10 → coincide con MAPEO_SECTOR
                (ej. códigos 11-19 → grupo 10, códigos 21-29 → grupo 20…).
    contrato_tipo: 'indefinido' | 'temporal'.
    """
    df = pd.read_csv('src/data/02_silver/train_silver/salarios_ocupacion.csv')
    df['grupo_isco'] = (df['ocupacion_isco08'] // 10).astype('Int64') * 10
    df['contrato_tipo'] = np.where(
        df['tipo_contrato'].str.lower().str.startswith('indefinido', na=False),
        'indefinido', 'temporal'
    )
    return (
        df.groupby(['grupo_isco', 'jornada', 'contrato_tipo'])['renta_neta_salarial']
        .median()
        .reset_index()
    )

_medianas_sal = _cargar_medianas_salario()

def salario_simulado(emp_id: str, ocupacion_isco08, antiguedad: int,
                     jornada: str = "Tiempo completo",
                     tipo_contrato: str = "Indefinido") -> float:
    """
    Estima la renta neta salarial anual usando la mediana real del CSV (ECV 2025),
    filtrada por grupo ISCO-08 (decena), jornada y tipo de contrato.
    Aplica una variación determinista ±10 % según emp_id y un ajuste por antigüedad.
    Si no hay datos para la combinación exacta, usa la mediana del grupo ISCO.
    """
    try:
        isco = int(float(ocupacion_isco08))
    except (ValueError, TypeError):
        isco = 91

    grupo = (isco // 10) * 10  # 11 → 10, 25 → 20, 91 → 90…

    jornada_key  = "Tiempo parcial" if "parcial" in str(jornada).lower() else "Tiempo completo"
    contrato_key = "temporal" if "temporal" in str(tipo_contrato).lower() else "indefinido"

    mask = (
        (_medianas_sal['grupo_isco']    == grupo) &
        (_medianas_sal['jornada']       == jornada_key) &
        (_medianas_sal['contrato_tipo'] == contrato_key)
    )
    fila = _medianas_sal.loc[mask, 'renta_neta_salarial']

    if fila.empty:
        # Fallback: mediana del grupo ISCO sin filtrar jornada/contrato
        fila = _medianas_sal.loc[_medianas_sal['grupo_isco'] == grupo, 'renta_neta_salarial']

    mediana = float(fila.iloc[0]) if not fila.empty else 18000.0

    # Variación determinista ±10 % según emp_id
    seed      = sum(ord(c) for c in str(emp_id)) % 100   # 0-99
    variacion = 1 + (seed - 50) / 500                    # 0.90 – 1.10
    # Ajuste por antigüedad: +0.8 % por año, máximo +15 %
    factor_antiguedad = 1 + min(int(antiguedad) * 0.008, 0.15)

    return round(mediana * variacion * factor_antiguedad, 0)
# ─────────────────────────────────────────────────────────────────────────────
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'rol' not in st.session_state:
    st.session_state.rol = None
if 'employee_id' not in st.session_state:
    st.session_state.employee_id = None
if 'consent_signed' not in st.session_state:
    st.session_state.consent_signed = False
if 'respuestas_guardadas' not in st.session_state:
    st.session_state.respuestas_guardadas = []

# ─────────────────────────────────────────────────────────────────────────────
# PANTALLA DE LOGIN
# ─────────────────────────────────────────────────────────────────────────────
def pantalla_login():
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown("""
        <div style="text-align:center; margin-bottom: 32px; padding-top: 40px;">
            <div style="font-family:'DM Serif Display',serif; font-size:2.4rem; color:#1b4b7a; font-weight:700;">
                FinWellsor
            </div>
            <div style="font-size:0.8rem; color:#94a3b8; letter-spacing:0.12em; text-transform:uppercase; margin-top:4px;">
                Bienestar Financiero · Plataforma Corporativa
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:white; border-radius:20px; padding:40px 36px; box-shadow:0 8px 40px rgba(0,0,0,0.10);">
            <div style="font-size:1.1rem; font-weight:600; color:#0f172a; margin-bottom:6px;">Bienvenido/a</div>
            <div style="font-size:0.88rem; color:#64748b; margin-bottom:28px;">
                Selecciona tu perfil de acceso para continuar
            </div>
        """, unsafe_allow_html=True)

        employee_id_input = st.text_input(
            "ID de empleado/a",
            placeholder="Ej: EMP-00142",
            help="Introduce tu número de empleado tal como aparece en tu nómina"
        )

        st.markdown("<div style='margin-top:16px; font-size:0.85rem; color:#475569; font-weight:500; margin-bottom:8px;'>Acceder como:</div>", unsafe_allow_html=True)

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("👤  Empleado/a", use_container_width=True, type="secondary"):
                if employee_id_input.strip():
                    st.session_state.logged_in = True
                    st.session_state.rol = 'empleado'
                    st.session_state.employee_id = employee_id_input.strip()
                    st.rerun()
                else:
                    st.error("Por favor introduce tu ID de empleado/a.")

        with col_btn2:
            if st.button("🏢  RRHH", use_container_width=True, type="primary"):
                if employee_id_input.strip():
                    st.session_state.logged_in = True
                    st.session_state.rol = 'rrhh'
                    st.session_state.employee_id = employee_id_input.strip()
                    st.rerun()
                else:
                    st.error("Por favor introduce tu ID de empleado/a.")

        st.markdown("""
            <div style="margin-top:28px; padding-top:20px; border-top:1px solid #f1f5f9; font-size:0.78rem; color:#94a3b8; text-align:center;">
                🔒 Acceso seguro · Tus datos están protegidos según el RGPD
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECCIONES COMPARTIDAS
# ─────────────────────────────────────────────────────────────────────────────
def seccion_contexto():
    rol = st.session_state.get('rol', 'empleado')

    if rol == 'rrhh':
        # Vista RRHH: métricas y texto orientado a la organización
        st.markdown('''
        <div class="hero-banner">
            <h1>Predicción de Estrés Financiero</h1>
            <p>Una herramienta de inteligencia artificial para que los departamentos de RRHH identifiquen, de forma proactiva, a los empleados en situación de vulnerabilidad económica.</p>
        </div>
        ''', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""<div class="metric-card danger">
                <div class="metric-value">1 de 6</div>
                <div class="metric-label">Asalariados madrileños con estrés financiero alto</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""<div class="metric-card warning">
                <div class="metric-value">< 20.000€</div>
                <div class="metric-label">Renta neta anual per cápita con mayor riesgo de estrés</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown("""<div class="metric-card info">
                <div class="metric-value">-30.000€/año</div>
                <div class="metric-label">Coste de rotación y pérdida de productividad</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown("""<div class="metric-card success">
                <div class="metric-value">+110%</div>
                <div class="metric-label">Retorno de la inversión (ROI) estimado</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        col_a, col_b = st.columns([3, 2], gap='large')
        with col_a:
            st.markdown('## El Problema')
            st.markdown('''
            El **estrés financiero** es uno de los principales factores ocultos de pérdida de rendimiento
            en las organizaciones. A diferencia de otros indicadores de bienestar, es difícil de detectar
            porque los empleados raramente lo comunican de forma directa.

            Las consecuencias son tangibles:
            - Mayor absentismo y rotación no deseada
            - Dificultad para concentrarse y tomar decisiones
            - Deterioro del clima laboral y de las relaciones de equipo
            - Costes de sustitución cuando el talento abandona la empresa

            Los departamentos de RRHH necesitan **anticiparse**, no reaccionar.
            ''')
            st.markdown('## La Solución')
            st.markdown('''
            Este modelo, entrenado con datos de la **Encuesta de Condiciones de Vida (ECV 2025)**,
            analiza variables socioeconómicas y laborales para predecir la probabilidad de que un
            empleado esté experimentando estrés financiero.

            No reemplaza el criterio humano de RRHH — lo potencia, permitiendo centrar los recursos
            de apoyo donde más se necesitan.
            ''')
        with col_b:
            st.markdown('## ¿Cómo funciona?')
            for i, paso in enumerate([
                ("Diagnóstico voluntario", "El empleado responde una encuesta breve sobre su situación. Sus respuestas son confidenciales y el proceso es completamente voluntario."),
                ("Evaluación inteligente", "La herramienta analiza automáticamente el perfil y estima la probabilidad de estrés financiero mediante un modelo de IA."),
                ("Resultado por niveles", "Se asigna un nivel de riesgo: Bajo, Medio o Alto, con recomendaciones para el equipo de RRHH."),
                ("Intervención focalizada", "RRHH diseña apoyos personalizados priorizando a quienes más lo necesitan."),
            ], 1):
                st.markdown(f'''
                <div style="display:flex; align-items:flex-start; margin-bottom:18px;">
                    <span class="step-num">{i}</span>
                    <div>
                        <strong style="color:#0d4f91;">{paso[0]}</strong><br>
                        <span style="color:#64748b; font-size:0.88rem;">{paso[1]}</span>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            st.markdown('<div class="info-box">📌 <strong>Fuente de datos:</strong> Encuesta de Condiciones de Vida (INE), edición 2025. Muestra representativa a nivel nacional.</div>', unsafe_allow_html=True)

    else:
        # Vista Empleado: métricas y texto orientados al bienestar personal
        st.markdown('''
        <div class="hero-banner">
            <h1>Tu Bienestar Financiero</h1>
            <p>Un espacio confidencial donde puedes compartir tu situación económica. Tu empresa quiere ayudarte con los recursos que más necesitas.</p>
        </div>
        ''', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""<div class="metric-card danger">
                <div class="metric-value">1 de 6</div>
                <div class="metric-label">Trabajadores en Madrid experimentan estrés financiero alto</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""<div class="metric-card warning">
                <div class="metric-value">76%</div>
                <div class="metric-label">De las personas con estrés financiero no lo comunican en el trabajo</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown("""<div class="metric-card info">
                <div class="metric-value">100%</div>
                <div class="metric-label">Confidencial — RRHH trabaja siempre con datos agregados y anónimos</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown("""<div class="metric-card success">
                <div class="metric-value">Gratis</div>
                <div class="metric-label">Todos los programas de apoyo disponibles son sin coste para ti</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        col_a, col_b = st.columns([3, 2], gap='large')
        with col_a:
            st.markdown('## ¿Por qué este cuestionario?')
            st.markdown('''
            El **estrés financiero** afecta al bienestar personal y a la calidad de vida de muchas personas,
            pero suele ser difícil de hablar abiertamente en el entorno laboral.

            Tu empresa quiere ser parte de la solución:
            - Orientación financiera personal y confidencial
            - Acceso anticipado a nómina sin coste
            - Talleres de gestión del presupuesto familiar
            - Apoyo psicológico vinculado al estrés económico
            - Mejoras en seguros y beneficios sociales

            **Cuéntanos cómo estás** para que podamos ofrecerte lo que realmente necesitas.
            ''')
            st.markdown('## Tu privacidad, garantizada')
            st.markdown('''
            Tus respuestas se vinculan únicamente a tu ID de empleado/a para evitar duplicidades.
            El equipo de RRHH trabaja siempre de forma **agregada y anónima**: nunca se tomarán
            decisiones individuales negativas basadas en este cuestionario.

            Participar es completamente **voluntario**. Puedes abandonar en cualquier momento.
            ''')
        with col_b:
            st.markdown('## ¿Qué ocurre cuando envías el formulario?')
            for i, paso in enumerate([
                ("Rellenas el cuestionario", "Respondes preguntas sobre tu situación económica y laboral. Todo de forma confidencial y voluntaria."),
                ("Tus datos llegan a RRHH", "El equipo de Recursos Humanos recibe la información de forma segura para analizar las necesidades de la plantilla."),
                ("RRHH diseña los apoyos", "Con los datos agregados, RRHH activa los programas de bienestar más adecuados para el conjunto de la plantilla."),
                ("Te contactamos si procede", "Si tu situación puede beneficiarse de algún recurso específico, RRHH se pondrá en contacto contigo de forma discreta."),
            ], 1):
                st.markdown(f'''
                <div style="display:flex; align-items:flex-start; margin-bottom:18px;">
                    <span class="step-num">{i}</span>
                    <div>
                        <strong style="color:#0d4f91;">{paso[0]}</strong><br>
                        <span style="color:#64748b; font-size:0.88rem;">{paso[1]}</span>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            st.markdown('<div class="info-box">🔒 <strong>Protección de datos:</strong> Tus respuestas están protegidas según el RGPD (Reglamento UE 2016/679) y la LOPDGDD. Tienes derecho de acceso, rectificación y supresión en cualquier momento.</div>', unsafe_allow_html=True)


def seccion_usuarios_valor():
    st.markdown("# Usuarios y Propuesta de Valor")
    st.markdown("*¿Quién usa esta herramienta y qué gana con ella?*")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        st.markdown("""
        <div style="background:#f8fafc; border-radius:14px; padding:28px; border-top:4px solid #0ea5e9;">
            <div style="font-size:2.4rem; margin-bottom:12px;">👩‍💼</div>
            <h3 style="margin-top:0;">Business Partner de RRHH</h3>
            <p style="color:#64748b; font-size:0.9rem;">Detectar colectivos en riesgo antes de que el problema se materialice en baja o renuncia.</p>
            <strong style="color:#0ea5e9; font-size:0.82rem; text-transform:uppercase;">Valor obtenido</strong>
            <p style="font-size:0.9rem; color:#19272e;">Vista global de riesgo por departamento o demografía. Soporte para decisiones de compensación y beneficios.</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:#f8fafc; border-radius:14px; padding:28px; border-top:4px solid #10b981;">
            <div style="font-size:2.4rem; margin-bottom:12px;">🧑‍⚕️</div>
            <h3 style="margin-top:0;">Técnico de Bienestar</h3>
            <p style="color:#64748b; font-size:0.9rem;">Identificar empleados que podrían beneficiarse de asistencia financiera o asesoramiento.</p>
            <strong style="color:#10b981; font-size:0.82rem; text-transform:uppercase;">Valor obtenido</strong>
            <p style="font-size:0.9rem; color:#19272e;">Lista priorizada de perfiles. Intervenciones dirigidas que maximizan el impacto de los recursos.</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background:#f8fafc; border-radius:14px; padding:28px; border-top:4px solid #f59e0b;">
            <div style="font-size:2.4rem; margin-bottom:12px;">📊</div>
            <h3 style="margin-top:0;">Director de RRHH</h3>
            <p style="color:#64748b; font-size:0.9rem;">Justificar la inversión en bienestar financiero con datos cuantificables.</p>
            <strong style="color:#f59e0b; font-size:0.82rem; text-transform:uppercase;">Valor obtenido</strong>
            <p style="font-size:0.9rem; color:#19272e;">ROI de los programas, reducción del coste de rotación y argumento objetivo para dirección general.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("## ¿Cómo encaja en el flujo de trabajo de RRHH?")
    col_left, col_right = st.columns([2, 1], gap="large")
    with col_left:
        for icono_titulo, desc in [
            ("🔵 Diagnóstico anual", "Se lanza la encuesta interna anónima una vez al año, alineada con la evaluación del clima laboral."),
            ("🟡 Análisis automático", "Los datos se procesan en esta plataforma. En minutos, RRHH tiene el mapa de riesgo de toda la plantilla."),
            ("🟠 Priorización", "El equipo identifica los colectivos de mayor riesgo (por área, contrato, tramo salarial…)."),
            ("🟢 Intervención", "Se activan programas específicos: asesoramiento financiero, anticipos de nómina, formación en finanzas personales."),
            ("🔁 Seguimiento", "Se mide el impacto en la siguiente edición. El modelo aprende con los nuevos datos."),
        ]:
            st.markdown(f"""
            <div style="display:flex; gap:16px; margin-bottom:14px; align-items:flex-start;">
                <div style="font-size:1.4rem; min-width:32px;">{icono_titulo.split()[0]}</div>
                <div>
                    <strong>{' '.join(icono_titulo.split()[1:])}</strong><br>
                    <span style="color:#64748b; font-size:0.88rem;">{desc}</span>
                </div>
            </div>""", unsafe_allow_html=True)
    with col_right:
        st.markdown("""
        <div class="warn-box">
            <span style="color:#1f4478;">
            <strong>⚖️ Privacidad y ética</strong><br><br>
            <span style="font-size:0.88rem; color:#64748b;">
            Esta herramienta está diseñada para ser usada de forma <strong>agregada y anónima</strong>.
            Las predicciones individuales solo deben activarse con consentimiento explícito del empleado.<br><br>
            El modelo es un <strong>apoyo a la decisión humana</strong>, nunca un sustituto del criterio de RRHH.
            </span>
        </div>""", unsafe_allow_html=True)


def seccion_eda():
    st.markdown("# Análisis Exploratorio de Datos")
    st.markdown("*Distribución y caracterización de la muestra ECV 2025*")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.info("📊 Esta sección mostrará los gráficos del análisis exploratorio una vez integrados los datos ECV 2025. Los paneles incluirán distribución de estrés por tramo salarial, tipo de contrato, composición del hogar y régimen de tenencia.")
    st.markdown("""
    **Variables más relevantes identificadas en el modelo:**
    - `renta_neta_hogar` y `renta_hogar_per_capita` — mayor poder predictivo
    - `ratio_carga_vivienda` — proporción del gasto en vivienda sobre ingresos
    - `carga_prestamos_no_vivienda` — deudas externas al hogar
    - `puede_vacaciones`, `puede_calefaccion_invierno` — indicadores de privación material
    - `tipo_contrato` y `jornada` — precariedad laboral
    """)

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN EDA
# ─────────────────────────────────────────────────────────────────────────────

def seccion_eda():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    st.markdown("# ¿Quién tiene más riesgo? Lo que nos dicen los datos")
    st.markdown("*Resumen de los principales hallazgos sobre la muestra analizada — Comunidad de Madrid, ECV 2025*")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        📋 <strong>¿De dónde vienen estos datos?</strong><br>
        El análisis se basa en la <strong>Encuesta de Condiciones de Vida 2025</strong> del INE,
        centrada en trabajadores asalariados de la Comunidad de Madrid.
        Los resultados reflejan patrones reales del mercado laboral madrileño y sirven
        de base para que el modelo de IA realice sus predicciones.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    # INSIGHT 1 — ¿Cuántos trabajadores están en riesgo?
    # ═══════════════════════════════════════════════════════════════
    st.markdown("## 👥 1 de cada 6 trabajadores madrileños tiene estrés financiero elevado")
    col_txt, col_chart = st.columns([2, 2], gap="large")

    with col_txt:
        st.markdown("""
        Según los datos analizados, aproximadamente el **17% de los asalariados**
        de la Comunidad de Madrid experimenta lo que el modelo considera estrés
        financiero alto: dificultades acumuladas para llegar a fin de mes, pagar
        facturas o cubrir gastos imprevistos.

        Aunque pueda parecer una proporción pequeña, en una empresa de 500 personas
        supone unas **85 personas** que probablemente estén lidiando con este problema
        de forma silenciosa.
        """)
        st.markdown("""
        <div class="metric-card danger" style="max-width:280px;">
            <div class="metric-value">15,8%</div>
            <div class="metric-label">de la plantilla podría estar en situación de riesgo financiero alto</div>
        </div>
        """, unsafe_allow_html=True)

    with col_chart:
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sizes  = [84.2, 15.8]
        colors = ['#d1fae5', '#fee2e2']
        wedges, texts, autotexts = ax.pie(
            sizes, labels=['Sin riesgo elevado\n(84,2%)', 'Riesgo financiero alto\n(15,8%)'],
            colors=colors, autopct='%1.0f%%', startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2},
            textprops={'fontsize': 9}
        )
        autotexts[0].set_color('#065f46')
        autotexts[1].set_color('#b91c1c')
        ax.set_title('Distribución del riesgo financiero\nen la muestra', fontsize=10, fontweight='bold', pad=12)
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    # INSIGHT 2 — Ingresos: el factor más determinante
    # ═══════════════════════════════════════════════════════════════
    st.markdown("## 💶 Los ingresos son el principal factor de riesgo")
    col_txt2, col_chart2 = st.columns([2, 2], gap="large")

    with col_txt2:
        st.markdown("""
        El análisis muestra una diferencia muy clara entre los ingresos de quienes
        tienen riesgo alto y quienes no:

        - Los trabajadores **sin riesgo** tienen una renta media salarial neta anual
          de aproximadamente **28.000€**.
        - Los trabajadores **con riesgo alto** se sitúan en torno a los **18.000 €**
          de renta media salarial neta anual.

        El umbral crítico se sitúa alrededor de los **20.000€ anuales per cápita en el hogar**:
        por debajo de esa cifra, el riesgo aumenta de forma notable.
        """)

    with col_chart2:
        fig2, ax2 = plt.subplots(figsize=(4.5, 3))
        categorias = ['Sin riesgo elevado', 'Riesgo alto']
        valores    = [28000, 18000]
        colores    = ['#6ee7b7', '#fca5a5']
        bars = ax2.barh(categorias, valores, color=colores, edgecolor='white', height=0.5)
        ax2.set_xlabel('Renta media salarial neta anual (€)', fontsize=8)
        ax2.set_title('Ingresos del trabajador/a según nivel de riesgo', fontsize=9, fontweight='bold')
        ax2.bar_label(bars, labels=[f'{v:,.0f} €' for v in valores], padding=5, fontsize=9, fontweight='bold')
        ax2.set_xlim(0, 38000)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        fig2.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    # INSIGHT 3 — Tipo de contrato y jornada
    # ═══════════════════════════════════════════════════════════════
    st.markdown("## 📋 El tipo de jornada marca la diferencia")

    col_chart3, col_txt3 = st.columns([2, 2], gap="large")

    with col_chart3:
        fig3, ax3 = plt.subplots(figsize=(3.5, 3))

        jornadas    = ['Completa', 'Parcial']
        pct_jornada = [14, 25]
        bars3 = ax3.bar(jornadas, pct_jornada, color=['#6ee7b7', '#fca5a5'], edgecolor='white', width=0.5)
        ax3.set_title('Riesgo alto por tipo de jornada', fontsize=9, fontweight='bold')
        ax3.set_ylabel('% con riesgo financiero alto', fontsize=8)
        ax3.set_ylim(0, 35)
        ax3.bar_label(bars3, labels=[f'{v}%' for v in pct_jornada], fontsize=10, fontweight='bold', padding=3)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
 
        fig3.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

    with col_txt3:
        st.markdown("""
        Los trabajadores con una **jornada parcial** tienen casi el doble de probabilidad de
        sufrir estrés financiero elevado que quienes tienen una jornada completa. 
        
        Aunque puede responder a elecciones personales, también está asociada a ingresos más bajos e 
        inestabilidad económica.

        **¿Qué implica para RRHH?**
        Las personas con jornadas reducidas son el grupo prioritario sobre el que actuar primero con 
        programas de apoyo.
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    # INSIGHT 4 — Vivienda y carga del hogar
    # ═══════════════════════════════════════════════════════════════
    st.markdown("## 🏠 El alquiler de mercado: el mayor factor de presión sobre el hogar")

    col_txt4, col_chart4 = st.columns([2, 2], gap="large")

    with col_txt4:
        st.markdown("""
        El **régimen de tenencia de la vivienda** tiene un impacto muy relevante
        en el nivel de estrés financiero:

        - Tanto los trabajadores que viven en **alquiler de mercado** como en **alquiler social** 
          son los grupos con mayor proporción de riesgo alto, especialmente cuando los ingresos son bajos.
        - Los propietarios **sin hipoteca** son el grupo con menor riesgo.
        - Tener **hipoteca** sitúa a las personas en una posición intermedia,
          ya que fija el gasto pero puede volverse pesada si los ingresos caen.

        Además, cuando el gasto mensual en vivienda supera el **40% de los ingresos**,
        el riesgo de estrés financiero se multiplica considerablemente.
        """)

    with col_chart4:
        fig4, ax4 = plt.subplots(figsize=(4.5, 3.2))
        tenencias = ['Propietario\nsin hipoteca', 'Cesión\ngratuita', 'Propietario\ncon hipoteca', 'Alquiler\nmercado', 'Alquiler\nsocial']
        pct_ten   = [8, 10, 16, 30, 35]
        color_ten = ['#6ee7b7', '#a7f3d0', '#fcd34d', '#fdba74', '#fca5a5']
        bars4 = ax4.barh(tenencias, pct_ten, color=color_ten, edgecolor='white', height=0.55)
        ax4.set_xlabel('% con riesgo financiero alto', fontsize=8)
        ax4.set_title('Riesgo según tipo de vivienda', fontsize=9, fontweight='bold')
        ax4.bar_label(bars4, labels=[f'{v}%' for v in pct_ten], padding=4, fontsize=8, fontweight='bold')
        ax4.set_xlim(0, 38)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        fig4.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    # INSIGHT 5 — Privación material: señales de alerta
    # ═══════════════════════════════════════════════════════════════
    st.markdown("## 🚨 Cuando no se puede permitir lo básico, el riesgo se dispara")

    col_chart5, col_txt5 = st.columns([2, 2], gap="large")

    with col_chart5:
        fig5, ax5 = plt.subplots(figsize=(4.5, 3))
        privaciones = ['No poder ir\nde vacaciones', 'No poder\ncalefactar\nel hogar', 'Ambas\na la vez']
        pct_priv    = [59, 53, 70]
        col_priv    = ['#fcd34d', '#f97316', '#ef4444']
        bars5 = ax5.bar(privaciones, pct_priv, color=col_priv, edgecolor='white', width=0.5)
        ax5.set_ylabel('% con riesgo financiero alto', fontsize=8)
        ax5.set_title('Riesgo según privaciones materiales declaradas', fontsize=8, fontweight='bold')
        ax5.set_ylim(0, 85)
        ax5.bar_label(bars5, labels=[f'{v}%' for v in pct_priv], fontsize=10, fontweight='bold', padding=3)
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        fig5.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig5, use_container_width=True)
        plt.close(fig5)

    with col_txt5:
        st.markdown("""
        Una de las señales más potentes que detecta el modelo es la **incapacidad
        de permitirse ciertas cosas básicas**:

        - Si alguien declara que **no puede calefactar su hogar en invierno**, hay un 53% de
          probabilidad de que tenga estrés financiero alto.
        - Si además **no puede permitirse vacaciones**, esa cifra sube al 59%.
        - Cuando se dan **ambas situaciones a la vez**, el riesgo alcanza el 70%.

        Estas preguntas forman parte del cuestionario que responden los empleados
        y son de las más valiosas para el modelo.
        """)
        st.markdown("""
        <div class="info-box">
            🔍 <strong>Para RRHH:</strong> Los empleados que señalan estas dificultades
            en el cuestionario deben ser los primeros en recibir información sobre
            los programas de apoyo disponibles.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("## 📌 Resumen: ¿qué perfil tiene más riesgo?")
    col_r1, col_r2, col_r3 = st.columns(3, gap="large")
    with col_r1:
        st.markdown("""
        <div style="background:#fef2f2; border-radius:14px; padding:20px; border-top:4px solid #ef4444;">
            <div style="font-size:1.5rem; margin-bottom:8px;">⚠️</div>
            <strong style="color:#0e47a1;">Perfil de mayor riesgo</strong>
            <ul style="font-size:0.88rem; color:#374151; margin-top:8px; padding-left:18px;">
                <li>Jornada parcial</li>
                <li>Renta neta anual por debajo de 18.000 €/año</li>
                <li>Vive de alquiler en el mercado libre</li>
                <li>No puede permitirse vacaciones ni calefacción</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with col_r2:
        st.markdown("""
        <div style="background:#fffbeb; border-radius:14px; padding:20px; border-top:4px solid #f59e0b;">
            <div style="font-size:1.5rem; margin-bottom:8px;">🔶</div>
            <strong style="color:#0e47a1;">Perfil de riesgo medio</strong>
            <ul style="font-size:0.88rem; color:#374151; margin-top:8px; padding-left:18px;">
                <li>Contrato indefinido pero con hipoteca elevada</li>
                <li>Renta neta anual entre 18.000 y 24.000 €/año</li>
                <li>Gasto en vivienda entre el 30% y 40% de ingresos</li>
                <li>Deudas adicionales fuera de la vivienda</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with col_r3:
        st.markdown("""
        <div style="background:#f0fdf4; border-radius:14px; padding:20px; border-top:4px solid #10b981;">
            <div style="font-size:1.5rem; margin-bottom:8px;">✅</div>
            <strong style="color:#0e47a1;">Perfil de bajo riesgo</strong>
            <ul style="font-size:0.88rem; color:#374151; margin-top:8px; padding-left:18px;">
                <li>Contrato indefinido a tiempo completo</li>
                <li>Renta familiar por encima de 29.000 €/año</li>
                <li>Propietario sin hipoteca o carga baja de vivienda</li>
                <li>Sin deudas significativas ni privaciones materiales</li>
            </ul>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FORMULARIO EMPLEADO (cuestionario completo)
# ─────────────────────────────────────────────────────────────────────────────
def seccion_cuestionario_empleado():
    emp_id = st.session_state.employee_id

    # ── BLOQUE: Voluntariedad e información previa ──
    st.markdown("# Cuestionario de Bienestar Financiero")
    st.markdown(f"*Sesión iniciada como empleado/a: **{emp_id}***")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="privacy-box">
        <div style="font-size:1.1rem; font-weight:700; color:#166534; margin-bottom:10px;">
            📋 Información importante antes de comenzar
        </div>
        <p style="color:#14532d; font-size:0.92rem; line-height:1.7; margin:0;">
            Este cuestionario es <strong>completamente voluntario</strong>. Nadie en la empresa está obligado a cumplimentarlo.<br><br>
            <strong>¿Para qué sirve?</strong> Los datos que proporciones permitirán al departamento de RRHH diseñar
            programas de apoyo al bienestar financiero adaptados a las necesidades reales de la plantilla: orientación financiera,
            acceso anticipado a nómina, talleres de finanzas personales o beneficios sociales adicionales.<br><br>
            <strong>¿Qué pasa con mis datos?</strong> Tus respuestas se vinculan únicamente a tu ID de empleado/a
            para evitar duplicidades. El equipo de RRHH trabaja siempre de forma <strong>agregada y anónima</strong>:
            nunca se tomarán decisiones individuales negativas basadas en este cuestionario.<br><br>
            Si en algún momento decides no continuar, puedes cerrar esta ventana sin consecuencias.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # BLOQUE: Consentimiento / firma digital
    st.markdown("## ✍️ Consentimiento y Protección de Datos")
    st.markdown("""
    <div class="consent-box">
        <strong>Cláusula de protección de datos (RGPD · Reglamento UE 2016/679)</strong><br><br>
        En cumplimiento del Reglamento General de Protección de Datos (RGPD) y de la Ley Orgánica 3/2018
        de Protección de Datos Personales y garantía de los derechos digitales (LOPDGDD), le informamos de que:<br><br>
        <strong>Responsable del tratamiento:</strong> [Nombre de la empresa] · DPO: dpo@empresa.com<br>
        <strong>Finalidad:</strong> Evaluación interna del bienestar financiero de la plantilla con fines de diseño de políticas de apoyo.<br>
        <strong>Base jurídica:</strong> Consentimiento explícito del interesado (Art. 6.1.a RGPD).<br>
        <strong>Destinatarios:</strong> Exclusivamente el equipo de RRHH de la organización. No se ceden datos a terceros.<br>
        <strong>Conservación:</strong> Los datos se conservarán durante el ciclo anual en curso y serán anonimizados transcurridos 12 meses.<br>
        <strong>Derechos:</strong> Puede ejercer sus derechos de acceso, rectificación, supresión, oposición y portabilidad
        escribiendo a dpo@empresa.com o mediante el formulario habilitado en el portal de empleados.<br><br>
        Al marcar la casilla inferior y proporcionar su firma, declara haber leído y comprendido esta información
        y presta su consentimiento libre, específico e informado para el tratamiento de sus datos.
    </div>
    """, unsafe_allow_html=True)

    col_consent1, col_consent2 = st.columns([2, 1])
    with col_consent1:
        acepta_rgpd = st.checkbox(
            "He leído y acepto la cláusula de protección de datos. Entiendo que mi participación es voluntaria y que puedo revocar este consentimiento en cualquier momento."
        )
        nombre_firma = st.text_input("Nombre completo (firma digital)", placeholder="Escribe tu nombre completo como firma")
    with col_consent2:
        fecha_firma = datetime.date.today().strftime("%d/%m/%Y")
        st.markdown(f"""
        <div class="firma-box">
            <div style="font-size:0.75rem; color:#94a3b8; margin-bottom:8px;">FECHA DE FIRMA</div>
            <div style="font-size:1.2rem; font-weight:700; color:#0f172a;">{fecha_firma}</div>
            <div style="font-size:0.75rem; color:#94a3b8; margin-top:8px;">ID EMPLEADO/A</div>
            <div style="font-size:1rem; color:#0ea5e9; font-weight:600;">{emp_id}</div>
        </div>
        """, unsafe_allow_html=True)

    consentimiento_valido = acepta_rgpd and nombre_firma.strip() != ""

    if not consentimiento_valido:
        st.warning("Por favor, acepta la cláusula de protección de datos y escribe tu nombre como firma para continuar con el cuestionario.")
        return

    st.session_state.consent_signed = True
    st.success("✅ Consentimiento registrado. Puedes continuar con el cuestionario.")

    # FORMULARIO
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("## Cuestionario")
    st.markdown('<div class="info-box">🔒 Los datos introducidos se vinculan a tu ID de empleado/a de forma confidencial. El equipo de RRHH solo accede a resultados agregados.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown("**📋 Datos personales**")
        edad = st.number_input("¿Cuál es tu edad?", min_value=16, max_value=90, value=30)
        sexo = st.selectbox("Sexo", ["Mujer", "Hombre"])
        estado_civil = st.selectbox("Estado civil", [
            "Casado/a o pareja de hecho", "Divorciado/a o separado/a", "Soltero/a", "Viudo/a"
        ])
        num_hijos = st.number_input("Número de hijos a cargo", min_value=0, max_value=10, value=0)
        nivel_educativo = st.selectbox("Nivel de estudios más alto completado", [
            "Sin estudios", "Educación primaria", "Educación secundaria", "Bachillerato o FP", "Universidad o superior"
        ])

    with col2:
        st.markdown("**💼 Situación laboral**")
        tipo_contrato = st.selectbox("Tipo de contrato", ["Indefinido", "Temporal"])
        horas_semana = st.number_input(
            "¿Cuántas horas trabajas a la semana (de media)?",
            min_value=0, max_value=60, step=5, value=40
        )
        tipo_jornada = st.selectbox("Tipo de jornada", ["Tiempo completo", "Tiempo parcial"])
        antiguedad = st.number_input(
            "¿Cuántos años llevas trabajando en total (vida laboral acumulada)?",
            min_value=0, max_value=50, value=5
        )
        expectativa_ingresos = st.selectbox(
            "¿Cómo crees que evolucionarán tus ingresos en los próximos 12 meses?",
            ["Mantenerse", "Mejorar", "Empeorar"]
        )
        cambio_ingresos = st.selectbox(
            "¿Cómo han cambiado los ingresos de tu hogar en los últimos 12 meses?",
            ["Han aumentado", "Se han mantenido igual", "Han disminuido"]
        )
        sector = st.selectbox("¿En qué categoría se encuadra mejor tu puesto de trabajo?", list(MAPEO_SECTOR.keys()))

    with col3:
        st.markdown("**🏠 Situación del hogar**")
        tipo_vivienda = st.selectbox("Régimen de tenencia de tu vivienda habitual", [
            "Propietario sin hipoteca", "Propietario con hipoteca", "Arrendatario", "Cesión gratuita", "Alquiler social"
        ])
        cuota_hipoteca = 0.0
        importe_alquiler = 0.0
        if tipo_vivienda == "Propietario con hipoteca":
            cuota_hipoteca = st.number_input("Cuota mensual de la hipoteca (€)", min_value=0, max_value=3000, step=50, value=600)
        if tipo_vivienda in ["Arrendatario", "Alquiler social"]:
            importe_alquiler = st.number_input("Importe mensual del alquiler (€)", min_value=0, max_value=4000, step=50, value=800)

        gastos_vivienda = st.number_input(
            "¿Cuánto gastas al mes en total en tu vivienda (hipoteca/alquiler + suministros + comunidad)? (€)",
            min_value=0, max_value=5000, value=900, step=50
        )
        renta_hogar = st.number_input(
            "¿Cuál es aproximadamente la renta neta mensual total de tu hogar (suma de todos los ingresos)? (€)",
            min_value=0, max_value=30000, value=2500, step=100
        )
        num_habitaciones = st.number_input(
            "¿Cuántas habitaciones (dormitorios) tiene tu vivienda?",
            min_value=0, max_value=10, value=2
        )
        num_miembros_mas14 = st.number_input(
            "¿Cuántas personas de 14 años o más (incluyéndote) vivís habitualmente en el hogar?",
            min_value=1, max_value=10, value=2
        )
        num_miembros_menos14 = st.number_input(
            "¿Cuántos niños menores de 14 años viven habitualmente en el hogar?",
            min_value=0, max_value=10, value=0
        )
        privacion_material = st.multiselect(
            "¿A cuál de las siguientes cosas no puedes hacer frente económicamente? (marca todas las que apliquen)",
            ["Vacaciones", "Comida (proteína mínima 2 días/semana)", "Reforma o sustitución de muebles", "Calefacción en invierno", "Ninguna de las anteriores"]
        )
        deudas = st.selectbox(
            "¿Tienes deudas activas (excluida la hipoteca)? ¿Cómo las valoras?",
            ["No", "Sí, son manejables", "Sí, me generan dificultades"]
        )
        dental = st.selectbox(
            "¿Cómo valoras la carga económica de los gastos de dentista en tu hogar?",
            ["No tengo gastos de dentista", "Los afronto sin problemas", "Son una carga importante"]
        )
        medicamentos = st.selectbox(
            "¿Cómo valoras la carga económica de los medicamentos y gastos sanitarios no cubiertos?",
            ["No tengo gastos de medicamentos", "Los afronto sin problemas", "Son una carga importante"]
        )

    st.markdown("---")
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        evaluar = st.button("🔍 Enviar y obtener resultado", type="primary", use_container_width=True)

    if evaluar:
        # Construcción del dict de features
        num_miembros_hogar = num_miembros_mas14 + num_miembros_menos14
        unidad_consumo = 1 + (0.5 * (num_miembros_mas14 - 1)) + (0.3 * num_miembros_menos14)

        privacion_norm = [p.replace("Comida (proteína mínima 2 días/semana)", "Comida").replace("Reforma o sustitución de muebles", "Reforma hogar").replace("Calefacción en invierno", "Calefacción") for p in privacion_material]
        respuesta_privacion = composicion_limitaciones_material(privacion_norm)

        tipo_hogar_calc = composicion_tipo_hogar(estado_civil, num_hijos, num_miembros_hogar, edad, sexo)

        # renta_neta_salarial la toma la empresa de su BBDD; aquí se usa la renta del hogar como proxy para la predicción
        # ya que el usuario no la introduce. Se marca como np.nan para reflejar que es dato empresarial.
        datos = {
            "edad": edad,
            "nivel_estudios": MAPEO_ESTUDIOS.get(nivel_educativo, np.nan),
            "tipo_contrato": MAPEO_CONTRATO.get(tipo_contrato, np.nan),
            "jornada": tipo_jornada,
            "horas_semana": horas_semana,
            "anios_experiencia": antiguedad,
            "renta_neta_salarial": salario_simulado(emp_id, MAPEO_SECTOR.get(sector, 91), antiguedad, tipo_jornada, tipo_contrato),  # simulado BBDD corporativa
            "expectativa_ingresos_12m": expectativa_ingresos,
            "ocupacion_isco08": MAPEO_SECTOR.get(sector, np.nan),
            "regimen_tenencia": MAPEO_VIVIENDA.get(tipo_vivienda, np.nan),
            "cuota_hipoteca": cuota_hipoteca,
            "importe_alquiler": importe_alquiler,
            "gastos_vivienda": gastos_vivienda,
            "renta_neta_hogar": renta_hogar * 12,
            "renta_hogar_per_capita": (renta_hogar * 12) / unidad_consumo if unidad_consumo > 0 else np.nan,
            "num_miembros_hogar": num_miembros_hogar,
            "carga_prestamos_no_vivienda": MAPEO_DEUDAS.get(deudas, np.nan),
            "tipo_hogar": tipo_hogar_calc,
            "puede_proteina_2dias": respuesta_privacion['puede_proteina_2dias'],
            "puede_vacaciones": respuesta_privacion['puede_vacaciones'],
            "puede_sustituir_muebles": respuesta_privacion['puede_sustituir_muebles'],
            "puede_calefaccion_invierno": respuesta_privacion['puede_calefaccion_invierno'],
            "ratio_carga_vivienda": (gastos_vivienda * 12) / (renta_hogar * 12) if renta_hogar > 0 else np.nan,
            "carga_asistencia_dental": MAPEO_DENTAL.get(dental, np.nan),
            "carga_medicamentos": MAPEO_MEDICAMENTOS.get(medicamentos, np.nan),
            "hogar_carencia_material": respuesta_privacion['hogar_carencia_material'],
            "cambio_ingresos_12m": MAPEO_CAMBIO_INGRESOS.get(cambio_ingresos, np.nan),
            "num_habitaciones": num_habitaciones,
            "unidades_consumo": unidad_consumo,
            "renta_no_monetaria_salarial": np.nan,  # dato corporativo — no disponible en cuestionario
        }

        score, label, css = predecir_individuo(datos)

        st.markdown(f"""
        <div class="prediction-result" style="background: linear-gradient(135deg, #f0fdf4, #d1fae5); border: 2px solid #6ee7b7;">
            <div style="font-size:3.5rem; margin-bottom:12px;">✅</div>
            <div style="font-size:2rem; font-family:'DM Serif Display',serif; font-weight:700; color:#065f46; margin-bottom:12px;">
                ¡Gracias por tu participación!
            </div>
            <div style="font-size:1rem; color:#047857; margin-bottom:16px; font-weight:500;">
                Tu formulario ha sido enviado a RRHH correctamente.
            </div>
            <div style="max-width:560px; margin:0 auto; font-size:0.92rem; color:#475569; background:rgba(255,255,255,0.7); border-radius:10px; padding:16px 22px; line-height:1.7;">
                El equipo de Recursos Humanos revisará tus respuestas de forma confidencial y, si lo considera oportuno, 
                se pondrá en contacto contigo para ofrecerte los recursos de bienestar más adecuados a tu situación.<br><br>
                Recuerda que tu participación es completamente <strong>voluntaria y confidencial</strong>. 
                Gracias por contribuir a mejorar el bienestar de toda la plantilla.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Generar CSV de respuestas
        salario_anual = salario_simulado(emp_id, MAPEO_SECTOR.get(sector, 91), antiguedad, tipo_jornada, tipo_contrato)
        fila_csv = {
            "id_empleado": emp_id,
            "nombre_firma": nombre_firma,
            "fecha_respuesta": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "score_riesgo": round(score, 4),
            "nivel_riesgo": label,
            "salario_neto_anual_simulado": salario_anual,
            "sector_label": sector,
        }
        fila_csv.update(datos)

        st.session_state.respuestas_guardadas.append(fila_csv)

        df_respuesta = pd.DataFrame([fila_csv])
        csv_bytes = df_respuesta.to_csv(index=False).encode('utf-8')

        st.markdown("---")
        st.markdown("### 📥 Descarga tu registro de respuestas")
        st.markdown("Puedes descargarte una copia de tus respuestas para tu archivo personal.")
        st.download_button(
            label="⬇️ Descargar mis respuestas (CSV)",
            data=csv_bytes,
            file_name=f"respuestas_bienestar_{emp_id}_{datetime.date.today()}.csv",
            mime="text/csv",
        )


# ─────────────────────────────────────────────────────────────────────────────
# HELPER DASHBOARD RRHH
# ─────────────────────────────────────────────────────────────────────────────
def _render_dashboard(df: pd.DataFrame):
    total = len(df)
    n_alto  = int((df['nivel_riesgo'] == 'Alto').sum())
    n_medio = int((df['nivel_riesgo'] == 'Medio').sum())
    n_bajo  = int((df['nivel_riesgo'] == 'Bajo').sum())
    sal_col = 'salario_neto_anual_simulado'

    # KPIs
    st.markdown("### 📊 Resumen de la plantilla")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(f"""<div class="metric-card info"><div class="metric-value">{total}</div>
        <div class="metric-label">Respuestas recibidas</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-card danger"><div class="metric-value">{n_alto}</div>
        <div class="metric-label">Riesgo Alto ({round(n_alto/total*100) if total else 0}%)</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="metric-card warning"><div class="metric-value">{n_medio}</div>
        <div class="metric-label">Riesgo Medio ({round(n_medio/total*100) if total else 0}%)</div></div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class="metric-card success"><div class="metric-value">{n_bajo}</div>
        <div class="metric-label">Riesgo Bajo ({round(n_bajo/total*100) if total else 0}%)</div></div>""", unsafe_allow_html=True)

    if sal_col in df.columns:
        df[sal_col] = pd.to_numeric(df[sal_col], errors='coerce')
        sal_alto = df.loc[df['nivel_riesgo'] == 'Alto', sal_col].mean()
        sal_total = df[sal_col].mean()
        sal_alto_fmt = f"{int(sal_alto):,} €" if pd.notna(sal_alto) else "—"
        c5.markdown(f"""<div class="metric-card danger"><div class="metric-value" style="font-size:1.5rem;">{sal_alto_fmt}</div>
            <div class="metric-label">Salario medio riesgo Alto</div></div>""", unsafe_allow_html=True)
    else:
        sal_total = None

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Gráficos
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("#### Distribución de riesgo")
        for nivel, cnt, color in zip(['Alto','Medio','Bajo'], [n_alto,n_medio,n_bajo], ['#ef4444','#f59e0b','#10b981']):
            pct = round(cnt / total * 100) if total else 0
            st.markdown(f"""<div style="margin-bottom:14px;">
                <div style="display:flex;justify-content:space-between;font-size:0.85rem;margin-bottom:4px;">
                    <span style="font-weight:600;color:{color};">{nivel}</span>
                    <span style="color:#64748b;">{cnt} empleados · {pct}%</span>
                </div>
                <div style="background:#f1f5f9;border-radius:6px;height:12px;">
                    <div style="background:{color};width:{pct}%;height:12px;border-radius:6px;"></div>
                </div></div>""", unsafe_allow_html=True)

        st.markdown("#### Score medio por nivel")
        score_medio = df.groupby('nivel_riesgo')['score_riesgo'].mean().reindex(['Alto','Medio','Bajo'])
        for nivel, color in zip(['Alto','Medio','Bajo'], ['#ef4444','#f59e0b','#10b981']):
            v = score_medio.get(nivel, np.nan)
            if pd.notna(v):
                st.markdown(f"""<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
                    <div style="width:12px;height:12px;border-radius:50%;background:{color};flex-shrink:0;"></div>
                    <span style="font-weight:600;color:{color};min-width:50px;">{nivel}</span>
                    <div style="flex:1;background:#f1f5f9;border-radius:4px;height:8px;">
                        <div style="background:{color};width:{round(v*100)}%;height:8px;border-radius:4px;"></div>
                    </div>
                    <span style="color:#475569;font-size:0.85rem;">{round(v*100,1)}/100</span>
                </div>""", unsafe_allow_html=True)

    with col_b:
        if sal_col in df.columns and sal_total:
            st.markdown("#### Salario neto anual medio por nivel (simulado)")
            sal_nivel = df.groupby('nivel_riesgo')[sal_col].mean().reindex(['Alto','Medio','Bajo'])
            sal_max = sal_nivel.max()
            for nivel, color in zip(['Alto','Medio','Bajo'], ['#ef4444','#f59e0b','#10b981']):
                v = sal_nivel.get(nivel, np.nan)
                if pd.notna(v):
                    pct_bar = round(v / sal_max * 100) if sal_max else 0
                    st.markdown(f"""<div style="margin-bottom:14px;">
                        <div style="display:flex;justify-content:space-between;font-size:0.85rem;margin-bottom:4px;">
                            <span style="font-weight:600;color:{color};">{nivel}</span>
                            <span style="color:#64748b;">{int(v):,} €/año</span>
                        </div>
                        <div style="background:#f1f5f9;border-radius:6px;height:12px;">
                            <div style="background:{color};width:{pct_bar}%;height:12px;border-radius:6px;"></div>
                        </div></div>""", unsafe_allow_html=True)

        if 'tipo_contrato' in df.columns:
            st.markdown("#### Tipo de contrato en riesgo Alto")
            df_alto = df[df['nivel_riesgo'] == 'Alto']
            if not df_alto.empty:
                for contrato, cnt in df_alto['tipo_contrato'].value_counts().items():
                    pct = round(cnt / len(df_alto) * 100)
                    st.markdown(f"""<div style="display:flex;justify-content:space-between;font-size:0.85rem;
                        padding:6px 0;border-bottom:1px solid #f1f5f9;">
                        <span style="color:#475569;">{contrato}</span>
                        <span style="font-weight:600;color:#ef4444;">{cnt} ({pct}%)</span>
                    </div>""", unsafe_allow_html=True)

        if 'expectativa_ingresos_12m' in df.columns:
            st.markdown("#### Expectativa ingresos (riesgo Alto)")
            df_alto = df[df['nivel_riesgo'] == 'Alto']
            if not df_alto.empty:
                for exp, cnt in df_alto['expectativa_ingresos_12m'].value_counts().items():
                    pct = round(cnt / len(df_alto) * 100)
                    color_exp = '#ef4444' if exp == 'Empeorar' else '#10b981' if exp == 'Mejorar' else '#f59e0b'
                    st.markdown(f"""<div style="display:flex;justify-content:space-between;font-size:0.85rem;
                        padding:6px 0;border-bottom:1px solid #f1f5f9;">
                        <span style="color:#475569;">{exp}</span>
                        <span style="font-weight:600;color:{color_exp};">{cnt} ({pct}%)</span>
                    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Tabla detalle riesgo Alto
    st.markdown("### 🔴 Empleados con riesgo Alto — detalle")
    df_alto_det = df[df['nivel_riesgo'] == 'Alto'].copy()
    if not df_alto_det.empty:
        cols_show = ['id_empleado', 'score_riesgo', 'nivel_riesgo']
        for extra in [sal_col, 'sector_label', 'edad', 'tipo_contrato', 'jornada',
                      'anios_experiencia', 'regimen_tenencia', 'ratio_carga_vivienda',
                      'hogar_carencia_material', 'carga_prestamos_no_vivienda', 'expectativa_ingresos_12m']:
            if extra in df_alto_det.columns:
                cols_show.append(extra)
        df_show = df_alto_det[[c for c in cols_show if c in df_alto_det.columns]].sort_values('score_riesgo', ascending=False).reset_index(drop=True)
        if sal_col in df_show.columns:
            df_show[sal_col] = df_show[sal_col].apply(lambda x: f"{int(x):,} €" if pd.notna(x) else "—")
        if 'ratio_carga_vivienda' in df_show.columns:
            df_show['ratio_carga_vivienda'] = df_show['ratio_carga_vivienda'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
        st.dataframe(df_show, use_container_width=True)
    else:
        st.success("✅ No hay empleados con nivel de riesgo Alto en esta muestra.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    csv_all = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Descargar todas las respuestas (CSV)", data=csv_all,
        file_name=f"respuestas_rrhh_{datetime.date.today()}.csv", mime="text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN RRHH: RESPUESTAS EMPLEADOS
# ─────────────────────────────────────────────────────────────────────────────
def seccion_respuestas_rrhh():
    st.markdown("# Panel de Respuestas de Empleados")
    st.markdown("*Respuestas recibidas en esta sesión y carga de ficheros externos*")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📋 Respuestas sesión actual", "📂 Cargar CSV externo y predecir"])

    with tab1:
        if st.session_state.respuestas_guardadas:
            df_resp = pd.DataFrame(st.session_state.respuestas_guardadas)
            st.success(f"✅ {len(df_resp)} respuesta(s) recibida(s) en esta sesión")
            _render_dashboard(df_resp)
        else:
            st.info("Todavía no hay respuestas registradas en esta sesión. Cuando un empleado complete el cuestionario, sus datos aparecerán aquí.")
            st.markdown("### 📄 Plantilla CSV (estructura de datos esperada)")
            ejemplo = pd.DataFrame([{
                "id_empleado": "EMP-00142", "score_riesgo": 0.23, "nivel_riesgo": "Bajo",
                "salario_neto_anual_simulado": 28560, "sector_label": "Técnico o perfil intermedio",
                "edad": 35, "tipo_contrato": "Indefinido escrito", "jornada": "Tiempo completo",
                "anios_experiencia": 8, "regimen_tenencia": "Propiedad con hipoteca",
                "ratio_carga_vivienda": 0.30, "hogar_carencia_material": "No",
                "expectativa_ingresos_12m": "Mantenerse",
            }])
            st.dataframe(ejemplo, use_container_width=True)
            csv_ej = ejemplo.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Descargar plantilla CSV", data=csv_ej, file_name="plantilla_respuestas.csv", mime="text/csv")

    with tab2:
        st.markdown("### Carga masiva de respuestas para predicción")
        st.markdown('<div class="warn-box">⚠️ El CSV debe contener las columnas del modelo. Usa la plantilla de la pestaña anterior como referencia.</div>', unsafe_allow_html=True)
        archivo = st.file_uploader("Sube el CSV con respuestas de empleados", type=["csv"])
        if archivo:
            df_carga = pd.read_csv(archivo, low_memory=False)
            st.markdown(f"**{len(df_carga)} registros cargados.** Vista previa:")
            st.dataframe(df_carga.head(10), use_container_width=True)
            if st.button("⚙️ Ejecutar predicción sobre todos los registros", type="primary"):
                df_resultado = predecir_batch(df_carga.copy())
                st.success("✅ Predicción completada")
                _render_dashboard(df_resultado)
                csv_res = df_resultado.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Descargar resultados con predicciones", data=csv_res,
                    file_name=f"predicciones_{datetime.date.today()}.csv", mime="text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN PRESUPUESTO (RRHH)
# ─────────────────────────────────────────────────────────────────────────────
def seccion_presupuesto():
    st.markdown("# Análisis de Presupuesto y ROI")
    st.markdown("*Justificación económica de la inversión en bienestar financiero*")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2], gap="large")
    with col_left:
        st.markdown("## Supuestos del modelo de ROI")
        plantilla = st.number_input("Tamaño de plantilla", min_value=50, max_value=50000, value=500, step=50)
        pct_riesgo = st.slider("% de empleados con riesgo Alto estimado", 5, 40, 17)
        coste_rotacion = st.number_input("Coste medio de reposición por baja (€)", min_value=5000, max_value=80000, value=30000, step=1000)
        pct_retencion = st.slider("% de bajas prevenibles con el programa", 10, 60, 25)
        coste_programa = st.number_input("Coste anual del programa de bienestar (€)", min_value=1000, max_value=500000, value=30000, step=1000)

        n_riesgo = int(plantilla * pct_riesgo / 100)
        n_bajas_evitadas = int(n_riesgo * pct_retencion / 100)
        ahorro_rotacion = n_bajas_evitadas * coste_rotacion
        roi = ((ahorro_rotacion - coste_programa) / coste_programa * 100) if coste_programa > 0 else 0

        st.markdown(f"""
        <table class="budget-table">
            <tr><th>Concepto</th><th>Valor</th></tr>
            <tr><td>Empleados en riesgo Alto estimados</td><td>{n_riesgo:,}</td></tr>
            <tr><td>Bajas evitadas con el programa</td><td>{n_bajas_evitadas:,}</td></tr>
            <tr><td>Ahorro por rotación evitada</td><td>{ahorro_rotacion:,.0f} €</td></tr>
            <tr><td>Coste del programa</td><td>{coste_programa:,.0f} €</td></tr>
            <tr><td>ROI estimado del programa</td><td>{roi:.0f}%</td></tr>
        </table>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("## ¿Qué incluye el programa?")
        st.markdown("""
        - 🧑‍💼 **Asesoramiento financiero individual** (externo, confidencial)
        - 📚 Talleres de **finanzas personales** y gestión del presupuesto familiar
        - 💳 Acceso a **anticipo de nómina** sin coste
        - 🏥 Mejora del **seguro de salud** o complemento dental
        - 📞 Línea de **atención psicológica** vinculada al estrés económico
        - 📊 Dashboard de seguimiento anual para RRHH
        """)
        st.markdown("""
        <div class="info-box">
            💡 <strong>Referencia de mercado:</strong> Según estudios de Mercer y PwC, cada euro invertido
            en programas de bienestar financiero genera entre 1,5 y 3 euros de retorno medible en productividad
            y reducción de rotación.
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN LIMITACIONES
# ─────────────────────────────────────────────────────────────────────────────
def seccion_limitaciones():
    st.markdown("# Limitaciones y Trabajo Futuro")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("## ⚠️ Limitaciones actuales")
        st.markdown("""
        **Fuente de datos:**
        - La ECV es una encuesta de diseño transversal: no permite inferir causalidad.
        - Muestra limitada a asalariados en Madrid; la generalización a otras regiones requiere validación.
        - La variable objetivo (`estres_financiero_alto`) es construida, no directamente declarada por el encuestado.

        **Modelo:**
        - LightGBM entrenado sobre datos históricos: puede no capturar shocks económicos recientes.
        - Las variables de renta se reportan con posibles sesgos de deseabilidad social.
        - Algunas variables clave (`renta_neta_salarial`) requieren integración con BBDD corporativa.

        **Uso:**
        - La herramienta no sustituye la evaluación clínica ni el diagnóstico profesional.
        - La predicción individual debe usarse únicamente en marcos de apoyo voluntario y con consentimiento.
        """)

    with col2:
        st.markdown("## 🚀 Líneas de trabajo futuro")
        st.markdown("""
        **Datos:**
        - Integración con la BBDD de nóminas para enriquecer automáticamente variables salariales.
        - Encuesta longitudinal interna: seguimiento año a año para medir evolución individual.
        - Ampliación de la muestra a otras comunidades autónomas.

        **Modelo:**
        - Exploración de modelos de series temporales para detectar deterioro progresivo.
        - Calibración del umbral de riesgo por sector económico.
        - Análisis de equidad algorítmica (fairness) por género, edad y origen.

        **Producto:**
        - Módulo de recomendaciones personalizadas para el empleado.
        - Integración con plataformas de RRHH (SAP SuccessFactors, Workday).
        - Dashboard en tiempo real para RRHH con alertas automáticas.
        """)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTING PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    pantalla_login()
else:
    rol = st.session_state.rol
    emp_id = st.session_state.employee_id

    with st.sidebar:
        st.markdown('<div class="sidebar-logo">FinWellsor</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sidebar-tagline">{"RRHH" if rol=="rrhh" else "Empleado/a"} · {emp_id}</div>', unsafe_allow_html=True)

        if rol == 'rrhh':
            opciones = [
                '🏠  Contexto',
                '👥  Usuarios y Valor',
                '📊  Análisis EDA',
                '💰  Presupuesto',
                '⚠️  Limitaciones y Futuro',
                '📋  Respuestas Empleados',
            ]
        else:
            opciones = [
                '🏠  Contexto',
                '📝  Cuestionario',
            ]

        seccion = st.radio('Navegación', options=opciones, label_visibility='collapsed')
        st.markdown('---')
        if rol == 'rrhh':
            estado_modelo = '✅ Modelo cargado' if modelo else '⚠️ Modelo no disponible (demo)'
            st.caption(estado_modelo)
        st.caption('TFM · ECV 2025')

        if st.button("🚪 Cerrar sesión", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.rol = None
            st.session_state.employee_id = None
            st.session_state.consent_signed = False
            st.rerun()

    # Routing de secciones
    if seccion == '🏠  Contexto':
        seccion_contexto()
    elif seccion == '👥  Usuarios y Valor':
        seccion_usuarios_valor()
    elif seccion == '📊  Análisis EDA':
        seccion_eda()
    elif seccion == '💰  Presupuesto':
        seccion_presupuesto()
    elif seccion == '⚠️  Limitaciones y Futuro':
        seccion_limitaciones()
    elif seccion == '📋  Respuestas Empleados':
        seccion_respuestas_rrhh()
    elif seccion == '📝  Cuestionario':
        seccion_cuestionario_empleado()
