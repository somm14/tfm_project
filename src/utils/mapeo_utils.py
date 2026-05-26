import numpy as np

# ----------------------------------------------------------------
# VARIABLES SELECCIONADAS PARA EL DATASET ANTES DE DESCODIFICARLAS
# ----------------------------------------------------------------
VARS_SELECCIONADAS = [

    # ── IDENTIFICADORES (necesarios para joins y trazabilidad) ────────────────
    'DB030',      # ID hogar
    'RB030',      # ID persona

    # ── PESO MUESTRAL ─────────────────────────────────────────────────────────
    'PB040',      # Factor de ponderación personal transversal

    # ── REGIÓN Y URBANIZACIÓN (ya filtrado Madrid, pero se conserva para EDA) ─
    'DB040',      # Región (ES30 = Madrid)
    'DB100',      # Grado de urbanización (zona muy/media/poco poblada)

    # ── DEMOGRÁFICAS PERSONA ──────────────────────────────────────────────────
    'RB080',      # Año de nacimiento (para calcular edad)
    'RB081',      # Edad a 31-dic año anterior
    'PB150',      # Sexo
    'RB280',      # País de nacimiento (proxy origen)
    'RB290',      # Nacionalidad

    # ── SITUACIÓN LABORAL ─────────────────────────────────────────────────────
    'PL032',      # Situación actividad autodefinida (filtro: =1 trabajando)
    'PL040A',     # Situación profesional empleo actual (filtro: =3 asalariado)
    'PL060',      # Horas trabajadas semana (intensidad laboral)
    'PL145',      # Jornada completa o parcial
    'PL141',      # Tipo de contrato: temporal (11,12) vs indefinido (21,22)
    'PL150',      # ¿Tiene personal a su cargo? (proxy responsabilidad/jerarquía)
    'PL200',      # Años de experiencia laboral remunerada
    'PL080',      # Meses en desempleo en año de referencia (historial desempleo)
    'PL271',      # Meses en desempleo en últimos 5 años (inestabilidad laboral)

    # ── OCUPACIÓN E INDUSTRIA ─────────────────────────────────────────────────
    'PL051A',     # Ocupación ISCO-08 (código numérico) # REVISAR: granularidad alta, considerar agrupar por grandes grupos
    'PL111AA',    # Sector actividad CNAE empleo actual # REVISAR: muy granular, agrupar si se usa

    # ── EDUCACIÓN ─────────────────────────────────────────────────────────────
    'PE041',      # Nivel de estudios terminados (ISCED)
    # PE010 y PE021 excluidos: estudios en curso → población activa laboralmente

    # ── SALUD ─────────────────────────────────────────────────────────────────
    'PH010',      # Estado general de salud (1=muy bueno → 5=muy malo)
    'PH020',      # ¿Enfermedad crónica? (puede limitar capacidad laboral)
    'PH030',      # Limitación por enfermedad (proxy discapacidad funcional)
    'PH040',      # ¿Necesitó médico y no fue? (proxy privación sanitaria)

    # ── RENTA INDIVIDUAL (asalariado) ─────────────────────────────────────────
    'PY010N',     # Renta neta monetaria del asalariado (variable continua clave)
    'PY020N',     # Renta neta no monetaria del asalariado (beneficios en especie)
    # PY010G excluido: renta bruta, redundante con la neta para el modelo

    # ── RENTA DEL HOGAR ───────────────────────────────────────────────────────
    'HY020',      # Renta disponible neta del hogar (variable continua clave)
    'vhRentaa',   # Renta hogar usada en indicadores armonizados de pobreza INE
    'HX240',      # Unidades de consumo (escala OCDE) → permite calcular renta/UC
    'HX040',      # Número de miembros del hogar
    'HX060',      # Tipo de hogar (composición familiar)
    # HY010 excluido: renta bruta hogar, redundante con HY020

    # ── INDICADORES DE POBREZA DEL INE (vars derivadas hogar) ────────────────
    'vhPobreza',  # Hogar en riesgo de pobreza (60% mediana renta/UC)
    'vhMATDEP',   # Hogar en carencia material severa (≥4 carencias de 9)

    # ── VARIABLES TARGET / COMPONENTES DEL ESTRÉS FINANCIERO ─────────────────
    'HS011',      # Retrasos hipoteca/alquiler últimos 12 meses
    'HS021',      # Retrasos pago facturas últimos 12 meses
    'HS031',      # Retrasos deudas no vivienda últimos 12 meses
    'HS060',      # Capacidad afrontar gastos imprevistos
    'HS120',      # Capacidad llegar a fin de mes (escala 1-6)

    # ── OTRAS VARIABLES DE PRIVACIÓN (componentes carencia material) ──────────
    'HS040',      # Puede permitirse vacaciones ≥1 semana/año
    'HS050',      # Puede permitirse proteína cada 2 días
    'HS090',      # Tiene ordenador
    'HS110',      # Tiene coche
    'HS150',      # Carga desembolsos préstamos no vivienda
    'HD080',      # Podría sustituir muebles estropeados
    'HH050',      # Puede mantener temperatura adecuada en invierno

    # ── VIVIENDA ──────────────────────────────────────────────────────────────
    'HH010',      # Tipo de vivienda (unifamiliar, piso, etc.)
    'HH021',      # Régimen de tenencia (propiedad, alquiler, cesión)
    'HH030',      # Número de habitaciones
    'HH060',      # Alquiler actual pagado (continua, solo si alquiler)
    'HH070',      # Gastos totales vivienda (alquiler/hipoteca + suministros)
    'cuotahip',   # Cuota hipoteca mensual (continua, solo si propiedad con hipoteca)

    # ── DINÁMICA DE INGRESOS ──────────────────────────────────────────────────
    'HI010',      # ¿Cambio en ingresos del hogar últimos 12 meses?
    'HI020',      # Motivo del aumento (solo si HI010=1)
    'HI030',      # Motivo de la disminución (solo si HI010=3)
    'HI040',      # Expectativa ingresos próximos 12 meses

    # ── CARGA FINANCIERA SANITARIA ────────────────────────────────────────────
    'HS200',      # Carga financiera asistencia médica
    'HS210',      # Carga financiera asistencia dental
    'HS220',      # Carga financiera medicamentos

    # ── VARIABLES DERIVADAS PERSONA (Fichero R) ────────────────────────────────
    'vrLOWJOB',       # Persona en hogar con baja intensidad laboral (Europa 2020)
    'vrEU2020',       # En riesgo pobreza o exclusión social (Europa 2020)
    'vrMATSOCDEP',    # En carencia material y social severa (≥7 de 13 carencias)
    'vrEU2030_nuevo', # En riesgo pobreza o exclusión social (Europa 2030)
    # vrLOWJOB_nuevo excluido: similar a vrLOWJOB con distinta definición de edad

    # ── FLAGS _F NECESARIOS PARA IMPUTACIÓN (se eliminan al final) ────────────
    'HB070_F',    # Flag informante
    'HB100_F',    # Flag minutos cuestionario
    'HS011_F', 'HS200_F', 'HS210_F',    'HS220_F', 'HS021_F', 'HS031_F',
    'HS150_F', 'HH060_F', 'cuotahip_F', 'HI020_F', 'PL060_F', 'PL271_F',
    'PH040_F', 'HI030_F', 'HS200_F',    'HS210_F', 'HS220_F'
]

# ----------------------------------------------------------------
# DESCODIFICACIÓN DEL NOMBRE DE LAS COLUMNAS
# ----------------------------------------------------------------

RENAME_MAP = {
    # Identificadores
    'DB030':          'id_hogar',
    'RB030':          'id_persona',
    'PB040':          'peso_persona',

    # Región y urbanización
    'DB040':          'region',
    'DB100':          'grado_urbanizacion',

    # Demográficas
    'RB080':          'anio_nacimiento',
    'RB081':          'edad',
    'PB150':          'sexo',
    'RB280':          'pais_nacimiento',
    'RB290':          'nacionalidad',

    # Situación laboral
    'PL032':          'situacion_actividad',
    'PL040A':         'situacion_profesional',
    'PL060':          'horas_semana',
    'PL145':          'jornada',
    'PL141':          'tipo_contrato',
    'PL150':          'personal_a_cargo',
    'PL200':          'anios_experiencia',
    'PL080':          'meses_desempleo_ref',
    'PL271':          'meses_desempleo_5anios',
    'PL051A':         'ocupacion_isco08',
    'PL111AA':        'sector_cnae',

    # Educación
    'PE041':          'nivel_estudios',

    # Salud
    'PH010':          'estado_salud',
    'PH020':          'enfermedad_cronica',
    'PH030':          'limitacion_actividad',
    'PH040':          'necesito_medico_no_fue',

    # Renta individual
    'PY010N':         'renta_neta_salarial',
    'PY020N':         'renta_no_monetaria_salarial',

    # Renta hogar
    'HY020':          'renta_neta_hogar',
    'vhRentaa':       'renta_hogar_indicadores',
    'HX240':          'unidades_consumo',
    'HX040':          'num_miembros_hogar',
    'HX060':          'tipo_hogar',

    # Indicadores pobreza hogar
    'vhPobreza':      'hogar_riesgo_pobreza',
    'vhMATDEP':       'hogar_carencia_material',

    # Estrés financiero (componentes target)
    'HS011':          'retrasos_hipoteca_alquiler',
    'HS021':          'retrasos_facturas',
    'HS031':          'retrasos_deudas_no_vivienda',
    'HS060':          'capacidad_gastos_imprevistos',
    'HS120':          'capacidad_fin_de_mes',

    # Privación material
    'HS040':          'puede_vacaciones',
    'HS050':          'puede_proteina_2dias',
    'HS090':          'tiene_ordenador',
    'HS110':          'tiene_coche',
    'HS150':          'carga_prestamos_no_vivienda',
    'HD080':          'puede_sustituir_muebles',
    'HH050':          'puede_calefaccion_invierno',

    # Vivienda
    'HH010':          'tipo_vivienda',
    'HH021':          'regimen_tenencia',
    'HH030':          'num_habitaciones',
    'HH060':          'importe_alquiler',
    'HH070':          'gastos_vivienda',
    'cuotahip':       'cuota_hipoteca',

    # Dinámica ingresos
    'HI010':          'cambio_ingresos_12m',
    'HI020':          'motivo_aumento_ingresos',
    'HI030':          'motivo_disminucion_ingresos',
    'HI040':          'expectativa_ingresos_12m',

    # Carga sanitaria
    'HS200':          'carga_asistencia_medica',
    'HS210':          'carga_asistencia_dental',
    'HS220':          'carga_medicamentos',

    # Variables derivadas persona
    'vrLOWJOB':       'baja_intensidad_laboral_2020',
    'vrEU2020':       'arope_2020',
    'vrMATSOCDEP':    'carencia_material_social_severa',
    'vrEU2030_nuevo': 'arope_2030',
}

# ----------------------------------------------------------------
# DESCODIFICACIONES DE LOS VALORES
# ----------------------------------------------------------------

DECODIFICACIONES = {

    'sexo': {'1': 'Hombre', '2': 'Mujer'},

    'grado_urbanizacion': {
        '1': 'Zona muy poblada', '2': 'Zona media', '3': 'Zona poco poblada'
    },

    'pais_nacimiento':  {'1': 'España', '2': 'UE', '3': 'Extranjero no UE'},
    'nacionalidad':     {'1': 'España', '2': 'UE', '3': 'Extranjero no UE'},

    'situacion_actividad': {
        '1': 'Trabajando', '2': 'Parado', '3': 'Jubilado', '4': 'Incapacitado',
        '5': 'Estudiante', '6': 'Labores hogar', '7': 'Sin cuestionario', '8': 'Otro inactivo'
    },

    'situacion_profesional': {
        '1': 'Empleador', '2': 'Autónomo sin asalariados', '3': 'Asalariado', '4': 'Ayuda familiar'
    },

    'jornada': {'1': 'Tiempo completo', '2': 'Tiempo parcial'},

    'tipo_contrato': {
        '11': 'Temporal escrito', '12': 'Temporal verbal',
        '21': 'Indefinido escrito', '22': 'Indefinido verbal'
    },

    'personal_a_cargo': {'1': 'Sí', '2': 'No'},

    # PL111AA: sector CNAE (letras, clasificación CNAE-2009 a nivel de sección)
    'sector_cnae': {
        'a': 'Agricultura/ganadería/pesca', 'b': 'Industrias extractivas',
        'c': 'Industria manufacturera', 'd': 'Energía eléctrica/gas',
        'e': 'Agua/saneamiento', 'f': 'Construcción',
        'g': 'Comercio/reparación vehículos', 'h': 'Transporte/almacenamiento',
        'i': 'Hostelería', 'j': 'Información/comunicaciones',
        'k': 'Actividades financieras/seguros', 'l': 'Actividades inmobiliarias',
        'm': 'Actividades profesionales/científicas', 'n': 'Actividades administrativas',
        'o': 'Administración pública/defensa', 'p': 'Educación',
        'q': 'Sanidad/servicios sociales', 'r': 'Actividades artísticas/recreativas',
        's': 'Otros servicios', 't': 'Hogares como empleadores',
        'u': 'Organismos extraterritoriales'
    },

    # HX060: tipo de hogar (composición familiar, 14 categorías)
    'tipo_hogar': {
        '1':  'Una persona: hombre <30 años', '2':  'Una persona: hombre 30-64 años',
        '3':  'Una persona: hombre ≥65 años', '4':  'Una persona: mujer <30 años',
        '5':  'Una persona: mujer 30-64 años', '6':  'Una persona: mujer ≥65 años',
        '7':  '2 adultos ≥65 sin niños', '8':  '2 adultos <65 sin niños',
        '9':  'Otros sin niños', '10': '1 adulto con niños',
        '11': '2 adultos, 1 niño', '12': '2 adultos, 2 niños',
        '13': '2 adultos, ≥3 niños', '14': 'Otros con niños'
    },


    'nivel_estudios': {
        '000': 'Sin estudios', '100': 'Primaria incompleta', '200': 'Primaria',
        '300': 'Secundaria 1ª etapa', '340': 'Secundaria 1ª etapa', '344': 'Secundaria 1ª etapa',
        '350': 'Secundaria 1ª etapa con título', '353': 'Secundaria 1ª etapa con título',
        '354': 'Secundaria 1ª etapa con título',
        '400': 'Secundaria 2ª etapa', '450': 'Secundaria 2ª etapa (orientación gral)',
        '500': 'Post-secundaria no superior', '600': 'Superior ciclo corto',
        '700': 'Grado/Licenciatura', '800': 'Máster', '900': 'Doctorado'
    },

    'estado_salud': {
        '1': 'Muy bueno', '2': 'Bueno', '3': 'Regular', '4': 'Malo', '5': 'Muy malo'
    },

    'enfermedad_cronica':       {'1': 'Sí', '2': 'No'},
    'limitacion_actividad': {
        '1': 'Gravemente limitado', '2': 'Limitado pero no gravemente', '3': 'No limitado'
    },
    'necesito_medico_no_fue':   {'1': 'Sí', '2': 'No'},

    'puede_vacaciones':         {'1': 'Sí', '2': 'No'},
    'puede_proteina_2dias':     {'1': 'Sí', '2': 'No'},
    'tiene_ordenador':          {'1': 'Sí', '2': 'No', '3': 'No (no puede permitírselo)'},
    'tiene_coche':              {'1': 'Sí', '2': 'No', '3': 'No (no puede permitírselo)'},
    'puede_sustituir_muebles':  {'1': 'Sí', '2': 'No (no puede permitírselo)', '3': 'No (otras razones)'},
    'puede_calefaccion_invierno': {'1': 'Sí', '2': 'No'},

    'retrasos_hipoteca_alquiler':  {'1': 'Sí, una vez', '2': 'Sí, dos o más veces', '3': 'No'},
    'retrasos_facturas':           {'1': 'Sí, una vez', '2': 'Sí, dos o más veces', '3': 'No'},
    'retrasos_deudas_no_vivienda': {'1': 'Sí, una vez', '2': 'Sí, dos o más veces', '3': 'No'},

    'capacidad_gastos_imprevistos': {'1': 'Sí', '2': 'No (no puede)', '3': 'No (otra razón)'},

    'capacidad_fin_de_mes': {
        '1': 'Con mucha dificultad', '2': 'Con dificultad', '3': 'Con cierta dificultad',
        '4': 'Con cierta facilidad', '5': 'Con facilidad', '6': 'Con mucha facilidad'
    },

    'carga_prestamos_no_vivienda': {
        '1': 'Una carga pesada', '2': 'Una carga razonable', '3': 'Ninguna carga'
    },

    'tipo_vivienda': {
        '1': 'Unifamiliar independiente', '2': 'Adosada/pareada',
        '3': 'Piso <10 viviendas', '4': 'Piso ≥10 viviendas'
    },

    'regimen_tenencia': {
        '1': 'Propiedad sin hipoteca', '2': 'Propiedad con hipoteca',
        '3': 'Alquiler precio mercado', '4': 'Alquiler precio reducido',
        '5': 'Cesión gratuita'
    },

    'cambio_ingresos_12m': {
        '1': 'Han aumentado', '2': 'Se mantienen', '3': 'Han disminuido'
    },

    'motivo_aumento_ingresos': {
        '1': 'Revalorización salario', '2': 'Más horas/salario', '3': 'Reincorporación trabajo',
        '4': 'Nuevo/cambio empleo', '5': 'Cambio composición hogar',
        '6': 'Prestaciones sociales', '7': 'Otros'
    },

    'motivo_disminucion_ingresos': {
        '1': 'Reducción horas/salario', '2': 'Baja parental/cuidados', '3': 'Cambio empleo',
        '4': 'Pérdida empleo', '5': 'Incapacidad', '6': 'Divorcio/separación',
        '7': 'Jubilación', '8': 'Reducción prestaciones', '9': 'Otros'
    },

    'expectativa_ingresos_12m': {
        '1': 'Mejorar', '2': 'Mantenerse', '3': 'Empeorar'
    },

    'carga_asistencia_medica': {
        '1': 'Una carga pesada', '2': 'Una carga razonable',
        '3': 'Ninguna carga', '4': 'No ha utilizado'
    },
    'carga_asistencia_dental': {
        '1': 'Una carga pesada', '2': 'Una carga razonable',
        '3': 'Ninguna carga', '4': 'No ha utilizado'
    },
    'carga_medicamentos': {
        '1': 'Una carga pesada', '2': 'Una carga razonable',
        '3': 'Ninguna carga', '4': 'No ha consumido'
    },

    'hogar_riesgo_pobreza':            {'0': 'No', '1': 'Sí'},
    'hogar_carencia_material':         {'0': 'No', '1': 'Sí'},
    'baja_intensidad_laboral_2020':    {'0': 'No', '1': 'Sí', '9': 'No aplicable (≥60 años)'},
    'arope_2020':                      {'0': 'No', '1': 'Sí'},
    'carencia_material_social_severa': {'0': 'No', '1': 'Sí'},
    'arope_2030':                      {'0': 'No', '1': 'Sí'},
}

# ----------------------------------------------------------------
# COMPONENTES DEL TARGET ESTRÉS FINANCIERO
# ----------------------------------------------------------------

COMPONENTES_ESTRES = {
    'capacidad_fin_de_mes':         ['Con mucha dificultad', 'Con dificultad'],
    'capacidad_gastos_imprevistos': ['No (no puede)'],
    'retrasos_facturas':            ['Sí, una vez', 'Sí, dos o más veces'],
    'retrasos_hipoteca_alquiler':   ['Sí, una vez', 'Sí, dos o más veces'],
    'retrasos_deudas_no_vivienda':  ['Sí, una vez', 'Sí, dos o más veces'],
}

# ----------------------------------------------------------------
# MAPEO PARA LA CATEGORIZACIÓN DEL NIVEL DE ESTUDIOS
# ----------------------------------------------------------------

MAPA_ESTUDIOS = {
    'Sin estudios':                 'Hasta primaria',
    'Primaria incompleta':          'Hasta primaria',
    'Primaria':                     'Hasta primaria',
    'Secundaria 1ª etapa':          'Secundaria 1a etapa',
    'Secundaria 1ª etapa (título)': 'Secundaria 1a etapa',
    'Secundaria 2ª etapa (gral)':   'Post-secundaria',
    'Post-secundaria no superior':  'Post-secundaria',
}

# ----------------------------------------------------------------
# ENCODING DE VARIABLES CATEGÓRICAS ORDINALES
# ----------------------------------------------------------------

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

# ----------------------------------------------------------------
# ENCODING DE VARIABLES CATEGÓRICAS BINARIAS
# ----------------------------------------------------------------

MAPA_BINARIO = {
    'Sí': 1.0, 'Si': 1.0, 'No': 0.0,
    'Hombre': 1.0, 'Mujer': 0.0,
    'Tiempo completo': 1.0, 'Tiempo parcial': 0.0,
    'No aplicable (≥60 años)': np.nan,
    'No aplicable (>=60 anos)': np.nan,
}