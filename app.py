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
    color: #5898db !important;
    font-size: 1.7rem !important;
}
h3 {
    font-family: 'DM Serif Display', sans-serif !important;
    font-weight: 600 !important;
    color: #5898db !important;
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
    model_path = Path('src/models/lg_streamlit.pkl')
    if model_path.exists():
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

cols_num_streamlit = ['edad',
 'anios_experiencia',
 'renta_neta_salarial',
 'cuota_hipoteca',
 'importe_alquiler',
 'gastos_vivienda',
 'renta_neta_hogar',
 'num_miembros_hogar',
 'ratio_carga_vivienda',
 'ocupacion_isco08',
 'renta_hogar_per_capita',
 'num_habitaciones',
 'unidades_consumo',
 'renta_no_monetaria_salarial',
 'horas_semana']

COLS_LOG1P = [
    'renta_neta_salarial', 'renta_no_monetaria_salarial', 'renta_neta_hogar',
    'renta_hogar_per_capita', 'importe_alquiler', 'cuota_hipoteca', 'gastos_vivienda',
]

IDX_LOG = [c for c in COLS_LOG1P if c in cols_num_streamlit]

def log1p_rentas(X):
    X = X.copy().astype(float)
    X[:, IDX_LOG] = np.log1p(np.clip(X[:, IDX_LOG], 0, None))
    return X

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

# ─────────────────────────────────────────────────────────────────────────────
# 3. PREDICCIONES
# ─────────────────────────────────────────────────────────────────────────────
elif seccion == "🔮  Predicciones":
    st.markdown("# Predicciones de Riesgo")
    st.markdown("*Evalúa el nivel de estrés financiero de uno o varios empleados*")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["👤  Evaluación Individual", "📂  Carga Masiva (CSV)"])

    # ── TAB 1: INDIVIDUAL ──
    with tab1:
        st.markdown("### Introduce el perfil del empleado")
        st.markdown('<div class="info-box">🔒 Los datos introducidos no se almacenan. La evaluación se realiza en tiempo real de forma local.</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3, gap="large")

        with col1:
            st.markdown("**📋 Datos personales**")
            edad = st.number_input("Edad",min_value=16, max_value=90, value=20)

            sexo = st.selectbox("Sexo", [
                "Mujer", "Hombre"
            ])
            estado_civil = st.selectbox("Estado Civil", [
                "Casado/a o pareja de hecho", "Divorciado/a o separado/a",
                "Soltero/a", "Viudo/a"
            ])
            num_hijos = st.number_input("Número de hijos a cargo", min_value=0, max_value=10, value=0)
            nivel_educativo = st.selectbox("Nivel educativo", [
                "Sin estudios", "Educación primaria", "Educación secundaria", "Bachillerato o FP", "Universidad o superior"
            ])

        with col2:
            st.markdown("**💼 Situación laboral**")
            tipo_contrato = st.selectbox("Tipo de contrato", [
                "Indefinido", "Temporal"
            ])
            if tipo_contrato:
                horas_semana = st.number_input(
                    "Introduzca las horas semanales de su jornada:",
                    min_value=0,
                    max_value=60,
                    step=10,
                    value=20 
                )

            tipo_jornada = st.selectbox("Tipo de jornada",[
                "Tiempo parcial", "Tiempo completo"
            ])
            antiguedad = st.number_input("Antigüedad laboral (vida laboral)", min_value=0, max_value=50, value=3)
            
            salario_neto = st.number_input("Salario neto mensual (€)", min_value=500, max_value=10000, value=1800, step=100)
            if salario_neto:
                expectativa_ingresos = st.selectbox(
                    "¿Cuáles son sus expectativas sobre sus ingresos de aquí a un año:",[
                        "Mantenerse", "Mejorar", "Empeorar"
                    ])

            sector = st.selectbox("Sector", [
                "Dirección y Gerencia", "Profesional cualificado (Ingeniería, tecnología, finanzas, leyes, etc.)", 
                "Técnico o perfil intermedio (Soporte, diseño, gestión media)",
                "Administración y oficina (Contabilidad, atención al cliente, secretariado)",
                "Ventas y Servicios (Comercial, atención en tienda, hostelería)",
                "Oficios manuales cualificados (Mecánica, mantenimiento, construcción)",
                "Operador de maquinaria o transporte (Conductor, reparto, operario)",
                "Tareas de apoyo o elementales (Limpieza, almacén, auxiliares)",
                "Sector agrícola o pesquero"
            ])

        with col3:
            st.markdown("**🏠 Situación del hogar**")
            tipo_hogar = st.selectbox("Régimen de vivienda", [
                "Propietario sin hipoteca", "Propietario con hipoteca", "Arrendatario", "Cesión gratuita", "Alquiler social"
            ])
            cuota_hipoteca = 0.0
            importe_alquiler = 0.0

            if tipo_hogar == "Propietario con hipoteca":
                cuota_hipoteca = st.number_input(
                    "Introduzca la cuota mensual de la hipoteca (€):",
                    min_value=0,
                    max_value=3000,
                    step=50,
                    value=500 
                )
            
            if tipo_hogar == "Arrendatario" or tipo_hogar == 'Alquiler social':
                cuota_hipoteca = st.number_input(
                    "Introduzca la cuota mensual del aquiler (€):",
                    min_value=0,
                    max_value=4000,
                    step=50,
                    value=500 
                )

            gastos_vivienda = st.number_input("Gasto mensual en vivienda (€)", min_value=0, max_value=3000, value=700, step=50)
            renta_hogar = st.number_input("Renta neta del hogar mensual (€)", min_value=0, max_value=30000, value=4000, step=50)

            privacion_material = st.multiselect("¿Tiene privación de algunos de estos materiales?", [
                "Vacaciones", "Comida", "Reforma hogar", "Calefacción", "Ninguna de las anteriores"
            ])

            num_miembros_mas14 = st.number_input("¿Cuántas personas de 14 años o más (incluyéndose a usted) viven habitualmente en su hogar?", min_value=1, max_value=10, value=2)
            num_miembros_menos14 = st.number_input("¿Cuántos niños menores de 14 años viven habitualmente en su hogar?", min_value=1, max_value=10, value=2)

            deudas = st.selectbox("¿Tiene deudas activas (excl. hipoteca)?", ["No", "Sí, manejables", "Sí, con dificultades"])

        st.markdown("---")
        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            evaluar = st.button("🔍 Evaluar riesgo", type="primary", use_container_width=True)

        if evaluar:
            # Construir dict de features (ajustar nombres a los del modelo real)
            mapeo_estudios = {
                "Bachillerato o FP": "post-secundaria",
                "Universidad o superior": "post-secundaria",
                "Educación primaria": "Hasta primaria",
                "Educación secundaria": "Secundaria 1a etapa"
            }
            estudios_mapeados = mapeo_estudios.get(nivel_educativo, np.nan)

            mapeo_contrato = {
                "Indefinido": "Indefinido escrito",
                "Temporal": "Temporal escrito"
            }
            contrato_mapeados = mapeo_contrato.get(tipo_contrato, np.nan)

            mapeo_vivienda = {
                "Propietario sin hipoteca": "Propiedad sin hipoteca",
                "Propietario con hipoteca": "Propiedad con hipoteca",
                "Arrendatario": "Alquiler precio mercado",
                "Cesión gratuita": "Cesión gratuita",
                "Alquiler social": "Alquiler precio reducido"
            }
            vivienda_mapeado = mapeo_vivienda.get(tipo_hogar, np.nan)

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
                
                elif estado_civil in ['Divorciado/a o separado/a', 'Soltero/a', "Viudo/a"] and num_hijos >= 1 and num_miembros >= 1:
                    return "1 adulto con niños"
                
                elif num_hijos == 0 and num_miembros == 1 and sexo == 'Hombre' and edad <= 30:
                    return "Una persona: hombre <30 años"
                
                elif num_hijos == 0 and num_miembros == 1 and sexo == 'Mujer' and edad <= 30:
                    return "Una persona: mujer <30 años"
        
                else:
                    return "Otro tipo de hogar"
                
            num_miembros_hogar = num_miembros_mas14 + num_miembros_menos14

            tipo_hogar_calculado = composicion_tipo_hogar(
                estado_civil=estado_civil, 
                num_hijos=num_hijos, 
                num_miembros=num_miembros_hogar,
                edad=edad, 
                sexo=sexo
            )

            
            def composicion_limitaciones_material(privacion_material):
                """
                Evalúa la lista de opciones seleccionadas en el multiselect
                y devuelve el estado de cada limitación material para el modelo.
                """
                resultado = {
                    'puede_proteina_2dias': 'Sí',
                    'puede_vacaciones': 'Sí',
                    'puede_sustituir_muebles': 'Sí',
                    'puede_calefaccion_invierno': 'Sí',
                    'hogar_carencia_material': 'No'
                }
                
                if not privacion_material or "Ninguna de las anteriores" in privacion_material:
                    return resultado  # Devuelve todo en 'Sí'
                    
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
            

            respuesta_privacion = composicion_limitaciones_material(privacion_material)

            mapeo_deudas = {
                "No": "Ninguna carga",
                "Sí, manejables": "Una carga razonable",
                "Sí, con dificultades": "Una carga pesada"
            }
            deudas_mapeado = mapeo_deudas.get(deudas, np.nan)

            # Cálculo de unidad de consumo por hogar

            unidad_consumo = 1 + (0.5 * (num_miembros_mas14 -1)) + (0.3 * num_miembros_menos14)


            mapeo_sector = {
                "Dirección y Gerencia": 11,
                "Profesional cualificado (Ingeniería, tecnología, finanzas, leyes, etc.)": 25,
                "Técnico o perfil intermedio (Soporte, diseño, gestión media)": 31,
                "Administración y oficina (Contabilidad, atención al cliente, secretariado)": 41,
                "Ventas y Servicios (Comercial, atención en tienda, hostelería)": 51,
                "Oficios manuales cualificados (Mecánica, mantenimiento, construcción)": 72,
                "Operador de maquinaria o transporte (Conductor, reparto, operario)": 83,
                "Tareas de apoyo o elementales (Limpieza, almacén, auxiliares)": 91,
                "Sector agrícola o pesquero": 61
            }

            sector_mapeado = mapeo_sector.get(sector, np.nan)

            datos = {
                "edad": edad,
                "nivel_estudios": estudios_mapeados,
                "tipo_contrato": contrato_mapeados,
                "jornada": tipo_jornada,
                "horas_semana": horas_semana,
                "anios_experiencia": antiguedad,
                "renta_neta_salarial": salario_neto * 12,
                "expectativa_ingresos_12m": expectativa_ingresos,
                "ocupacion_isco08": sector_mapeado,
                "regimen_tenencia": vivienda_mapeado,
                "cuota_hipoteca": cuota_hipoteca,
                "importe_alquiler": importe_alquiler,
                "gastos_vivienda": gastos_vivienda,
                "renta_neta_hogar": renta_hogar * 12,
                "renta_hogar_per_capita": (renta_hogar * 12) / unidad_consumo,
                "num_miembros_hogar": num_miembros_hogar,
                "carga_prestamos_no_vivienda": deudas_mapeado,
                "tipo_hogar": tipo_hogar_calculado,
                "puede_proteina_2dias": respuesta_privacion['puede_proteina_2dias'],
                "puede_vacaciones": respuesta_privacion['puede_vacaciones'],
                "puede_sustituir_muebles": respuesta_privacion['puede_sustituir_muebles'],
                "puede_calefaccion_invierno": respuesta_privacion['puede_calefaccion_invierno'],
                "ratio_carga_vivienda": gastos_vivienda * 12 / salario_neto,
                'carga_asistencia_dental': np.nan, 
                'carga_medicamentos': np.nan,
                'hogar_carencia_material':respuesta_privacion['hogar_carencia_material'], 
                'cambio_ingresos_12m': np.nan,
                'num_habitaciones': np.nan,
                'unidades_consumo': unidad_consumo,
                'renta_no_monetaria_salarial': np.nan
            }

            if modelo:
                score, label, css = predecir_individuo(datos)
            else:
                # Simulación demo si no hay modelo
                score = round(np.random.beta(2, 3), 4)
                label, css = nivel_riesgo(score)

            emoji_map = {"Alto": "🔴", "Medio": "🟡", "Bajo": "🟢"}
            consejo_map = {
                "Alto": "Se recomienda contacto proactivo del equipo de bienestar. Este perfil presenta múltiples factores de vulnerabilidad que requieren atención prioritaria.",
                "Medio": "Perfil en zona de vigilancia. Se recomienda incluir en el próximo ciclo de encuesta de bienestar y revisar si existe acceso a los programas de apoyo disponibles.",
                "Bajo": "Perfil con bajo riesgo en el momento actual. Se recomienda mantener el seguimiento en el ciclo anual habitual.",
            }

            st.markdown(f"""
            <div class="prediction-result pred-{css}">
                <div style="font-size:3rem; margin-bottom:8px;">{emoji_map[label]}</div>
                <div style="font-size:1rem; color:#64748b; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:4px;">Nivel de riesgo</div>
                <div style="font-size:3rem; font-family:'DM Serif Display',serif; font-weight:700; color:#0f1923; margin-bottom:8px;">{label}</div>
                <div style="font-size:1.4rem; color:#475569; margin-bottom:16px;">Puntuación: <strong>{round(score*100,1)}/100</strong></div>
                <div style="max-width:520px; margin:0 auto; font-size:0.92rem; color:#475569; background:rgba(255,255,255,0.6); border-radius:10px; padding:14px 20px;">
                    {consejo_map[label]}
                </div>
            </div>
            """, unsafe_allow_html=True)