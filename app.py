'''
app.py - Demo TFM: Predicción de Estrés Financiero
---------------------------------------------------
Interfaz de negocio para RRHH. Ejecutar con:
    streamlit run app.py
'''

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
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

/* Base */
html, body, [class*='css'] {
    font-family: 'DM Sans', sans-serif;
}

/* Sidebar */
[data-testid='stSidebar'] {
    background: linear-gradient(180deg, #0f1923 0%, #1a2d3d 100%);
    border-right: 1px solid #2a3f52;
}
[data-testid='stSidebar'] * {
    color: #e8edf2 !important;
}
[data-testid='stSidebar'] .stRadio label {
    font-size: 0.9rem;
    padding: 6px 0;
    letter-spacing: 0.02em;
}

/* Encabezados */
h1 {
    font-family: 'DM Serif Display', serif !important;
    color: #506478 !important;
    font-size: 2.4rem !important;
    letter-spacing: -0.01em;
}
h2 {
    font-family: 'DM Serif Display', serif !important;
    color: #3c4e61 !important;
    font-size: 1.7rem !important;
}
h3 {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    color: #1a2d3d !important;
}

/* Cards métricas */
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 24px 20px;
    border-left: 4px solid #0ea5e9;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    margin-bottom: 8px;
}
.metric-card.danger { border-left-color: #ef4444; }
.metric-card.warning { border-left-color: #f59e0b; }
.metric-card.success { border-left-color: #10b981; }
.metric-card.info { border-left-color: #0ea5e9; }

.metric-value {
    font-size: 2.2rem;
    font-weight: 600;
    color: #0f1923;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.82rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

/* Sección hero */
.hero-banner {
    background: linear-gradient(135deg, #0f1923 0%, #1a4060 60%, #0ea5e9 100%);
    border-radius: 16px;
    padding: 48px 40px;
    margin-bottom: 32px;
    color: white;
}
.hero-banner h1 {
    color: white !important;
    font-size: 2.8rem !important;
    margin-bottom: 8px;
}
.hero-banner p {
    color: #b0d4e8;
    font-size: 1.1rem;
    max-width: 600px;
}

/* Tags de riesgo */
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-alto { background: #fee2e2; color: #b91c1c; }
.badge-medio { background: #fef3c7; color: #92400e; }
.badge-bajo { background: #d1fae5; color: #065f46; }

/* Resultado predicción */
.prediction-result {
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    margin: 24px 0;
}
.pred-alto { background: linear-gradient(135deg, #fef2f2, #fee2e2); border: 2px solid #fca5a5; }
.pred-medio { background: linear-gradient(135deg, #fffbeb, #fef3c7); border: 2px solid #fcd34d; }
.pred-bajo { background: linear-gradient(135deg, #f0fdf4, #d1fae5); border: 2px solid #6ee7b7; }

/* Tabla CSV resultados */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* Separador sección */
.section-divider {
    height: 3px;
    background: linear-gradient(90deg, #0ea5e9, transparent);
    border-radius: 2px;
    margin: 24px 0 32px 0;
}

/* Info box */
.info-box {
    background: #1d3240;
    border: 1px solid #bae6fd;
    border-radius: 10px;
    padding: 18px 20px;
    margin: 16px 0;
}
.warn-box {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 10px;
    padding: 18px 20px;
    margin: 16px 0;
}

/* Steps */
.step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: #0ea5e9;
    color: white;
    border-radius: 50%;
    font-weight: 700;
    font-size: 0.9rem;
    margin-right: 10px;
}

/* Presupuesto tabla */
.budget-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}
.budget-table th {
    background: #0f1923;
    color: white;
    padding: 12px 16px;
    text-align: left;
    font-weight: 500;
    letter-spacing: 0.04em;
}
.budget-table td {
    padding: 11px 16px;
    border-bottom: 1px solid #e5e7eb;
}
.budget-table tr:nth-child(even) td { background: #f8fafc; }
.budget-table tr:last-child td {
    font-weight: 700;
    background: #f0f9ff;
    border-top: 2px solid #0ea5e9;
}

/* Sidebar logo */
.sidebar-logo {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: white;
    margin-bottom: 4px;
}
.sidebar-tagline {
    font-size: 0.72rem;
    color: #7faec2;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 28px;
}
</style>
''', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CARGA DEL MODELO (cacheado)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    model_path = Path('src/models/lighgbm_grid.pkl')
    if model_path.exists():
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

modelo = cargar_modelo()

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
FEATURES_DEL = [
    'motivo_aumento_ingresos', 'motivo_disminucion_ingresos',
    'id_hogar', 'id_persona', 'region',
    'capacidad_fin_de_mes', 'capacidad_gastos_imprevistos',
    'retrasos_facturas', 'retrasos_hipoteca_alquiler',
    'retrasos_deudas_no_vivienda',
    'estres_financiero_alto', 'peso_persona',
]

def nivel_riesgo(score: float) -> tuple[str, str]:
    if score >= 0.70:
        return 'Alto', 'alto'
    elif score >= 0.40:
        return 'Medio', 'medio'
    else:
        return 'Bajo', 'bajo'

def predecir_individuo(data: dict) -> tuple[float, str, str]:
    if modelo is None:
        return 0.0, 'Sin modelo', 'bajo'
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
# SIDEBAR NAVEGACIÓN
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">FinWellsor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Bienestar Financiero · RRHH</div>', unsafe_allow_html=True)

    seccion = st.radio(
        'Navegación',
        options=[
            '🏠  Contexto',
            '👥  Usuarios y Valor',
            '🔮  Predicciones',
            '📊  Análisis EDA',
            '💰  Presupuesto',
            '⚠️  Limitaciones y Futuro',
        ],
        label_visibility='collapsed',
    )

    st.markdown('---')
    estado_modelo = '✅ Modelo cargado' if modelo else '⚠️ Modelo no disponible'
    st.caption(estado_modelo)
    st.caption('TFM · ECV 2025')

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONTEXTO
# ─────────────────────────────────────────────────────────────────────────────
if seccion == '🏠  Contexto':
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
            <div class="metric-value"> Menos 24.000€ pp </div>
            <div class="metric-label">Rentas netas anuales con riesgo de estrés</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card info">
            <div class="metric-value">-30.000€/año</div>
            <div class="metric-label">Coste de Rotación y Pérdida de Productividad</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card success">
            <div class="metric-value">+110%</div>
            <div class="metric-label">Retorno de la Inversión (ROI)                          </div>
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
            ("Diagnóstico anónimo", "El empleado responde una encuesta breve sobre su situación laboral, económica y de vivienda. Sus respuestas son completamente confidenciales."),
            ("Evaluación inteligente", "La herramienta analiza automáticamente el perfil y estima la probabilidad de que el empleado esté experimentando estrés financiero."),
            ("Resultado por niveles", "Cada empleado recibe una valoración de riesgo: Bajo, Medio o Alto, con una explicación de los factores más relevantes en su caso."),
            ("Intervención focalizada", "RRHH puede diseñar apoyos personalizados — desde orientación financiera hasta beneficios sociales — priorizando a quienes más lo necesitan."),
        ], 1):

            st.markdown(f'''
            <div style='display:flex; align-items:flex-start; margin-bottom:18px;'>
                <span class='step-num'>{i}</span>
                <div>
                    <strong style='color:##3e4a57;'>{paso[0]}</strong><br>
                    <span style='color:#64748b; font-size:0.88rem;'>{paso[1]}</span>
                </div>
            </div>
            ''', unsafe_allow_html=True)

        st.markdown('<div class="info-box">📌 <strong>Fuente de datos:</strong> Encuesta de Condiciones de Vida (INE), edición 2025. Muestra representativa a nivel nacional.</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. USUARIOS
# ─────────────────────────────────────────────────────────────────────────────
elif seccion == "👥  Usuarios y Valor":
    st.markdown("# Usuarios y Propuesta de Valor")
    st.markdown("*¿Quién usa esta herramienta y qué gana con ella?*")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown("""
        <div style="background:#f8fafc; border-radius:14px; padding:28px; border-top:4px solid #0ea5e9; height:100%;">
            <div style="font-size:2.4rem; margin-bottom:12px;">👩‍💼</div>
            <h3 style="margin-top:0;">Business Partner de RRHH</h3>
            <p style="color:#64748b; font-size:0.9rem; margin-bottom:16px;">Perfil estratégico con visión global de plantilla</p>
            <strong style="color:#0ea5e9; font-size:0.82rem; text-transform:uppercase; letter-spacing:0.06em;">Necesidad</strong>
            <p style="font-size:0.9rem; color:#19272e;">Detectar colectivos en riesgo antes de que el problema se materialice en baja o renuncia.</p>
            <strong style="color:#0ea5e9; font-size:0.82rem; text-transform:uppercase; letter-spacing:0.06em;">Valor obtenido</strong>
            <p style="font-size:0.9rem; color:#19272e;">Vista global de riesgo por departamento o demografía. Soporte para decisiones de compensación y beneficios.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background:#f8fafc; border-radius:14px; padding:28px; border-top:4px solid #10b981; height:100%;">
            <div style="font-size:2.4rem; margin-bottom:12px;">🧑‍⚕️</div>
            <h3 style="margin-top:0;">Técnico de Bienestar</h3>
            <p style="color:#64748b; font-size:0.9rem; margin-bottom:16px;">Responsable de programas de salud y bienestar laboral</p>
            <strong style="color:#10b981; font-size:0.82rem; text-transform:uppercase; letter-spacing:0.06em;">Necesidad</strong>
            <p style="font-size:0.9rem; color:#19272e;">Identificar empleados que podrían beneficiarse de asistencia financiera o asesoramiento.</p>
            <strong style="color:#10b981; font-size:0.82rem; text-transform:uppercase; letter-spacing:0.06em;">Valor obtenido</strong>
            <p style="font-size:0.9rem; color:#19272e;">Lista priorizada de perfiles. Intervenciones dirigidas que maximizan el impacto de los recursos disponibles.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background:#f8fafc; border-radius:14px; padding:28px; border-top:4px solid #f59e0b; height:100%;">
            <div style="font-size:2.4rem; margin-bottom:12px;">📊</div>
            <h3 style="margin-top:0;">Director de RRHH</h3>
            <p style="color:#64748b; font-size:0.9rem; margin-bottom:16px;">Visión ejecutiva y responsabilidad de presupuesto</p>
            <strong style="color:#f59e0b; font-size:0.82rem; text-transform:uppercase; letter-spacing:0.06em;">Necesidad</strong>
            <p style="font-size:0.9rem; color:#19272e;">Justificar la inversión en bienestar financiero con datos cuantificables.</p>
            <strong style="color:#f59e0b; font-size:0.82rem; text-transform:uppercase; letter-spacing:0.06em;">Valor obtenido</strong>
            <p style="font-size:0.9rem; color:#19272e;">ROI de los programas, reducción del coste de rotación y argumento objetivo para la dirección general.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("## ¿Cómo encaja en el flujo de trabajo de RRHH?")

    col_left, col_right = st.columns([2, 1], gap="large")
    with col_left:
        flujo = [
            ("🔵 Diagnóstico anual", "Se lanza la encuesta interna anónima una vez al año, alineada con el proceso de evaluación del clima laboral."),
            ("🟡 Análisis automático", "Los datos se procesan en esta plataforma. En minutos, RRHH tiene el mapa de riesgo de toda la plantilla."),
            ("🟠 Priorización", "El equipo identifica los colectivos de mayor riesgo (por área, contrato, tramo salarial…)."),
            ("🟢 Intervención", "Se activan programas específicos: asesoramiento financiero, anticipos de nómina, formación en finanzas personales."),
            ("🔁 Seguimiento", "Se mide el impacto en la siguiente edición. El modelo aprende con los nuevos datos."),
        ]
        for icono_titulo, desc in flujo:
            st.markdown(f"""
            <div style="display:flex; gap:16px; margin-bottom:14px; align-items:flex-start;">
                <div style="font-size:1.4rem; min-width:32px;">{icono_titulo.split()[0]}</div>
                <div>
                    <strong>{' '.join(icono_titulo.split()[1:])}</strong><br>
                    <span style="color:#64748b; font-size:0.88rem;">{desc}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="warn-box">
            <strong>⚖️ Privacidad y ética</strong><br><br>
            <span style="font-size:0.88rem; color:#64748b;">
            Esta herramienta está diseñada para ser usada de forma <strong>agregada y anónima</strong>.
            Las predicciones individuales solo deben activarse con consentimiento explícito del empleado
            y en el marco de un programa de apoyo voluntario.<br><br>
            El modelo es un <strong>apoyo a la decisión humana</strong>, nunca un sustituto del criterio del equipo de RRHH.
            </span>
        </div>
        """, unsafe_allow_html=True)
