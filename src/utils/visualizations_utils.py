import numpy as np
import os
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from matplotlib.patches import FancyBboxPatch

from pathlib import Path

from scipy import stats

from utils.constants_utils import PALETTE, C_NEUTRAL, C0, C1
from utils.functions_utils import tasa_estres_cat

os.chdir(Path(__file__).resolve().parent.parent.parent)

plt.rcParams.update({
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'font.family': 'sans-serif',
})



def nuls_space_dis(df, target):
    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    fig.suptitle('Distribución en bruto de los 5 componentes del estrés financiero\n'
                '(antes de cualquier limpieza)', fontweight='bold', y=1.02)

    # Filtrar temporalmente solo Madrid + asalariados para el plot (sin ninguna limpieza más)
    df_pre = df[target].copy()
    
    for ax, col in zip(axes, target):
        vc = df_pre[col].fillna('NaN real').replace('', '[vacío/nulo]').replace(' ', '[vacío/nulo]')
        vc = vc.value_counts().head(6)
        colors = ['#C73E1D' if '[vacío' in str(k) or k=='NaN real' else '#2E86AB' for k in vc.index]
        ax.barh(vc.index.astype(str), vc.values, color=colors)
        ax.set_title(col, fontweight='bold')
        ax.set_xlabel('n hogares')

    plt.tight_layout()
    plt.show()
    print('Nota: las barras rojas representan valores faltantes según el INE (cadena vacía).')



def diagram_relationship():
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis('off')

    ficheros = [
        ('01\ndatos_hogar\n(D)', 29369, 9,  0.12, '#2E86AB'),
        ('02\ndetalles_hogar\n(H)', 29369, 156, 0.37, '#A23B72'),
        ('04\ndatos_persona\n(R)', 71398, 46,  0.62, '#F18F01'),
        ('03\ndetalles_adulto\n(P)', 60825, 245, 0.87, '#3B7D4F'),
    ]

    for label, filas, cols, x, color in ficheros:
        ax.add_patch(FancyBboxPatch((x-0.10, 0.2), 0.19, 0.6,
            boxstyle='round,pad=0.02', facecolor=color, alpha=0.85, edgecolor='white', lw=2))
        ax.text(x, 0.68, label, ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        ax.text(x, 0.30, f'{filas:,} filas\n{cols} cols', ha='center', va='center', fontsize=8, color='white')

    # flechas join
    ax.annotate('', xy=(0.30, 0.5), xytext=(0.10, 0.50),
        arrowprops=dict(arrowstyle='->', color='black', lw=1.8))
    ax.text(0.20, 0.55, 'DB030=HB030', ha='center', fontsize=7.5, color='black')

    ax.annotate('', xy=(0.88, 0.50), xytext=(0.67, 0.50),
        arrowprops=dict(arrowstyle='<-', color='black', lw=1.8))
    ax.text(0.77, 0.55, 'RB030=PB030', ha='center', fontsize=7.5, color='black')

    ax.annotate('', xy=(0.62-0.01, 0.38), xytext=(0.37+0.01, 0.38),
        arrowprops=dict(arrowstyle='<->', color='#555', lw=1.5, linestyle='dashed'))
    ax.text(0.495, 0.28, 'RB030//100 = DB030', ha='center', fontsize=7.5, color='#555')

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title('Estructura de ficheros ECV 2025 y relaciones de join', fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig('./src/img/bronze_diagrama_ficheros.png', bbox_inches='tight')
    plt.show()



def urban_grade_CCMM(data):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(data.index, data.values, color=PALETTE[:3])
    ax.bar_label(bars, fmt='%d%%', padding=4, fontsize=9)
    ax.set_xlabel('Número de personas')
    ax.set_title('Grado de urbanización en la muestra de Madrid\n'
                '(todos los residentes, antes de filtrar asalariados)', fontweight='bold')
    ax.set_xlim(0, data.max() * 1.15)
    plt.tight_layout()
    plt.savefig('./src/img/silver_data_urbanizacion.png', bbox_inches='tight')
    plt.show()
    print("-> La Comunidad de Madrid es eminentemente urbana: el 81% reside en zonas muy pobladas.")



def PL032_vs_PL040A(df):
    activ_map = {
    '1': 'Trabajando', '2': 'Parado', '3': 'Jubilado',
    '4': 'Incapacitado', '5': 'Estudiante', '6': 'Labores hogar', '8': 'Otro inactivo'
    }
    pl032_vc = df['PL032'].astype(str).str.strip().map(activ_map).value_counts(normalize=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Gráfico 1: Situación de actividad
    bars1 = axes[0].barh(pl032_vc.index, pl032_vc.values, color=PALETTE)
    axes[0].bar_label(bars1, fmt='%.2f%%', padding=3, fontsize=8)
    axes[0].set_xlabel('Personas')
    axes[0].set_title('PL032 - Situación de actividad\n(muestra Madrid, N=6.035)', fontweight='bold')
    axes[0].set_xlim(0, pl032_vc.max() * 1.2)

    # Gráfico 2: Situación profesional entre quienes trabajan
    df_trab = df[df['PL032'].astype(str).str.strip() == '1']
    prof_map = {'1': 'Empleador', '2': 'Autónomo', '3': 'Asalariado', '4': 'Ayuda familiar'}
    prof_vc = df_trab['PL040A'].astype(str).str.strip().map(prof_map).value_counts(normalize=True) * 100
    prof_vc = prof_vc[prof_vc.index.notna()]
    bars2 = axes[1].bar(prof_vc.index, prof_vc.values,
                        color=['#C73E1D' if k=='Asalariado' else '#ccc' for k in prof_vc.index])
    axes[1].bar_label(bars2, fmt='%.2f%%', padding=3, fontsize=8)
    axes[1].set_ylabel('Personas')
    axes[1].set_title('PL040A - Situación profesional\n(solo quienes trabajan: PL032=1)', fontweight='bold')

    plt.tight_layout()
    plt.savefig('./src/img/silver_filtros_laborales.png', bbox_inches='tight')
    plt.show()



def dis_target_descod(df):
    target_cols = ['capacidad_fin_de_mes', 'capacidad_gastos_imprevistos',
               'retrasos_facturas', 'retrasos_hipoteca_alquiler', 'retrasos_deudas_no_vivienda']

    orden_fin_mes = ['Con mucha dificultad', 'Con dificultad', 'Con cierta dificultad',
                    'Con cierta facilidad', 'Con facilidad', 'Con mucha facilidad']

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Componentes del estrés financiero - distribución tras decodificación\n'
                f'(asalariados de Madrid, N={len(df)})', fontweight='bold', fontsize=13, y=1.01)
    axes = axes.flatten()

    colores_stress = {
        'Con mucha dificultad': '#C73E1D', 'Con dificultad': '#F18F01',
        'Con cierta dificultad': '#F5C842',
        'Con cierta facilidad': '#8CBB6E', 'Con facilidad': '#4A9B6E',
        'Con mucha facilidad': '#2E7D52',
        'No (no puede)': '#C73E1D', 'No (otra razón)': '#F18F01', 'Sí': '#2E7D52',
        'Sí, una vez': '#F18F01', 'Sí, dos o más veces': '#C73E1D', 'No': '#2E7D52',
    }

    for i, col in enumerate(target_cols):
        ax = axes[i]
        vc = df[col].value_counts(dropna=False)
        if col == 'capacidad_fin_de_mes':
            cats_orden = [c for c in orden_fin_mes if c in vc.index]
            vc = vc[cats_orden]
        colors_bar = [colores_stress.get(str(k), '#999') for k in vc.index]
        bars = ax.barh([str(k) for k in vc.index], vc.values, color=colors_bar)
        ax.bar_label(bars, fmt='%d', padding=3, fontsize=8)
        pct_nulos = df[col].isna().sum() / len(df) * 100
        titulo = col.replace('_', '\n')
        ax.set_title(f'{titulo}\n({pct_nulos:.2f}% NaN)', fontsize=9, fontweight='bold')
        ax.set_xlabel('n personas')
        ax.set_xlim(0, vc.max() * 1.25)

    axes[5].axis('off')
    plt.tight_layout()
    plt.savefig('./src/img/silver_target_distribucion.png', bbox_inches='tight')
    plt.show()



def dis_bar_nuls(df):
    null_pct = df.isnull().mean() * 100
    null_pct = null_pct[null_pct > 0].sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors_bar = ['#C73E1D' if v > 50 else '#F18F01' if v > 20 else '#2E86AB' for v in null_pct.values]
    bars = ax.bar(range(len(null_pct)), null_pct.values, color=colors_bar, width=0.7)
    ax.set_xticks(range(len(null_pct)))
    ax.set_xticklabels(null_pct.index, rotation=45, ha='right', fontsize=8)
    ax.axhline(50, color='#C73E1D', linestyle='--', alpha=0.6, label='50%')
    ax.axhline(20, color='#F18F01', linestyle='--', alpha=0.6, label='20%')
    ax.set_ylabel('% de valores nulos')
    ax.set_title('Porcentaje de nulos por variable - dataset analítico Silver\n'
                '(solo variables con algún nulo; nulos estructurales incluidos)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)

    for i, (col, pct) in enumerate(zip(null_pct.index, null_pct.values)):
        if pct > 5:
            ax.text(i, pct + 1.5, f'{pct:.0f}%', ha='center', fontsize=6.5, rotation=90)

    plt.tight_layout()
    plt.savefig('./src/img/silver_mapa_nulos.png', bbox_inches='tight')
    plt.show()

    print("\nNota: los nulos más elevados (HI030, HI020) son ESTRUCTURALES.")
    print("No responden a errores de datos sino a que la pregunta no aplica al perfil del encuestado.")



def distribucion_target(y_train, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, (nombre, y_ser) in zip(axes, [('Train', y_train), ('Test', y_test)]):
        vc = y_ser.value_counts().sort_index()
        bars = ax.bar(['0 - Sin estrés', '1 - Estrés alto'],
                    vc.values,
                    color=['#2E86AB', '#C73E1D'],
                    width=0.5, edgecolor='white')
        ax.bar_label(bars, fmt='%d', padding=4, fontsize=10)
        for i, (val, n) in enumerate(zip(vc.index, vc.values)):
            ax.text(i, n / 2, f'{n/len(y_ser)*100:.1f}%', ha='center', va='center',
                    color='white', fontsize=11, fontweight='bold')
        ax.set_title(f'{nombre}  (n={len(y_ser):,})', fontweight='bold')
        ax.set_ylabel('Personas')
        ax.set_ylim(0, vc.max() * 1.15)

    fig.suptitle('Distribución del target tras stratify - la proporción 84/16% se mantiene',
                fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('src/img/gold_split_target.png', bbox_inches='tight')
    plt.show()

    # Verificación cuantitativa de la estratificación
    ratio_train = (y_train == 1).mean()
    ratio_test  = (y_test  == 1).mean()
    print(f'Ratio clase 1 - Train: {ratio_train:.4f}  |  Test: {ratio_test:.4f}')
    print(f'Diferencia absoluta:   {abs(ratio_train - ratio_test):.4f}  (debe ser < 0.005)')



def dis_components_target(df, componentes_estres):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle('Distribución de los 5 componentes del estrés financiero',
                fontweight='bold', y=1.01)

    COLORES_COMP = {'stress': '#C73E1D', 'ok': '#2E86AB', 'nan': '#999'}

    for ax, (col, vals_stress) in zip(axes, componentes_estres.items()):
        vc = df[col].value_counts(dropna=False)
        colores = []
        for k in vc.index:
            if pd.isna(k):          colores.append(COLORES_COMP['nan'])
            elif k in vals_stress:  colores.append(COLORES_COMP['stress'])
            else:                   colores.append(COLORES_COMP['ok'])
        bars = ax.barh([str(k)[:25] for k in vc.index], vc.values, color=colores)
        ax.bar_label(bars, fmt='%d', padding=3, fontsize=8)
        pct_stress = df[col].isin(vals_stress).mean() * 100
        ax.set_title(col.replace('_', '\n'), fontsize=8, fontweight='bold')
        ax.set_xlabel(f'n  ({pct_stress:.1f}% estrés)', fontsize=8)
        ax.set_xlim(0, vc.max() * 1.2)

    plt.tight_layout()
    plt.savefig('src/img/gold_componentes_target.png', bbox_inches='tight')
    plt.show()
    print('Rojo = condición activa | Azul = sin estrés | Gris = NaN')



def ratio_carga_vivienda(df):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    df['ratio_carga_vivienda'].dropna().hist(bins=50, ax=ax, color='#2E86AB', alpha=0.8)
    ax.axvline(0.30, color='#C73E1D', linestyle='--', lw=1.5, label='30% (umbral sobrecarga)')
    ax.set_xlabel('Ratio gastos vivienda / renta salarial neta')
    ax.set_ylabel('Personas')
    ax.set_title('Distribución del ratio de carga de vivienda', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig('src/img/gold_ratio_vivienda.png', bbox_inches='tight')
    plt.show()

    n_sob = (df['ratio_carga_vivienda'] > 0.30).sum()
    print(f'Personas con ratio > 30% (sobrecarga): {n_sob} ({n_sob/len(df)*100:.1f}%)')



def distribucion_logs(df, cols_log):
    fig, axes = plt.subplots(2, len(cols_log), figsize=(18, 6))
    fig.suptitle('Efecto de log1p sobre las distribuciones de renta y vivienda', fontweight='bold')

    for i, col in enumerate(cols_log):
        if col not in df.columns: continue
        datos = df[col].dropna()
        axes[0, i].hist(datos, bins=40, color='#2E86AB', alpha=0.8, edgecolor='white')
        axes[0, i].set_title(f'{col}\nskew={datos.skew():.2f}', fontsize=7)
        axes[0, i].set_ylabel('n' if i == 0 else '')
        datos_log = np.log1p(datos)
        axes[1, i].hist(datos_log, bins=40, color='#A23B72', alpha=0.8, edgecolor='white')
        axes[1, i].set_title(f'log1p({col})\nskew={datos_log.skew():.2f}', fontsize=7)
        axes[1, i].set_ylabel('n' if i == 0 else '')

    plt.tight_layout()
    plt.savefig('src/img/gold_log1p_rentas.png', bbox_inches='tight')
    plt.show()



def nulls_map(nulos, pct):
    fig, ax = plt.subplots(figsize=(8, max(3, len(nulos) * 0.4)))
    bars = ax.barh(nulos.index[::-1], pct.values[::-1], color=C_NEUTRAL, alpha=0.8)
    ax.bar_label(bars, [f'{v}%' for v in pct.values[::-1]], padding=3, fontsize=8)
    ax.set_xlabel('% nulos en train')
    ax.set_title('Variables con nulos en train set', fontweight='bold')
    ax.axvline(5, color='#C73E1D', linestyle='--', lw=1, alpha=0.7, label='5%')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('src/img/eda_mapa_nulos.png', bbox_inches='tight')
    plt.show()



def target_dis(total, n0, n1):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Barras de conteo
    bars = axes[0].bar(['0 - Sin estrés', '1 - Estrés alto'], [n0, n1],
                    color=[C0, C1], width=0.5, edgecolor='white')
    axes[0].bar_label(bars, fmt='%d', padding=4)
    for i, n in enumerate([n0, n1]):
        axes[0].text(i, n / 2, f'{n/total*100:.1f}%', ha='center', va='center',
                    color='white', fontsize=13, fontweight='bold')
    axes[0].set_title('Distribución muestral del target', fontweight='bold')
    axes[0].set_ylabel('Personas en train')

    # Gráfico de tarta
    axes[1].pie(
        [n0, n1],
        labels=[f'0 - Sin estrés\n({n0/total*100:.1f}%)', f'1 - Estrés alto\n({n1/total*100:.1f}%)'],
        colors=[C0, C1], autopct='%1.1f%%', startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
    )
    axes[1].set_title('Target: estres_financiero_alto', fontweight='bold')

    fig.suptitle(f'Desbalanceo 1:{n0/n1:.1f} - se gestionará con class_weight="balanced"',
                fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('src/img/eda_target.png', bbox_inches='tight')
    plt.show()

    print(f'Clase 0: {n0:,}  ({n0/total*100:.1f}%)')
    print(f'Clase 1: {n1:,}  ({n1/total*100:.1f}%)')
    print(f'Ratio de desbalanceo: 1:{n0/n1:.1f}')



def target_age_dis(t0, t1, train):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Distribución por clase
    axes[0].hist(t0['edad'].dropna(), bins=30, color=C0, alpha=0.6,
                label='Sin estrés', density=True)
    axes[0].hist(t1['edad'].dropna(), bins=30, color=C1, alpha=0.6,
                label='Estrés alto', density=True)
    axes[0].set_title('Distribución de edad por clase', fontweight='bold')
    axes[0].set_xlabel('Edad')
    axes[0].set_ylabel('Densidad')
    axes[0].legend(fontsize=9)

    # Tasa de estrés por tramo de edad
    bins_edad   = [16, 25, 35, 45, 55, 65]
    labels_edad = ['16-24', '25-34', '35-44', '45-54', '55-64']
    train['_tramo_edad'] = pd.cut(train['edad'], bins=bins_edad,
                                labels=labels_edad, right=False)
    tasa_edad = tasa_estres_cat(train, '_tramo_edad')
    axes[1].bar(tasa_edad.index.astype(str), tasa_edad.values,
                color=C1, alpha=0.85, edgecolor='white')
    axes[1].set_title('Tasa de estrés por tramo de edad', fontweight='bold')
    axes[1].set_xlabel('Tramo de edad')
    axes[1].set_ylabel('% estrés alto')
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))

    # Boxplot
    axes[2].boxplot([t0['edad'].dropna(), t1['edad'].dropna()],
                    labels=['Sin estrés', 'Estrés alto'],
                    patch_artist=True,
                    boxprops=dict(facecolor='white'),
                    medianprops=dict(color='black', lw=2))
    axes[2].set_title('Boxplot edad por clase', fontweight='bold')
    axes[2].set_ylabel('Edad')

    plt.tight_layout()
    plt.savefig('src/img/eda_edad.png', bbox_inches='tight')
    plt.show()
    train = train.drop(columns=['_tramo_edad'])

    print(f'Edad media - sin estrés: {t0["edad"].mean():.1f}  |  estrés alto: {t1["edad"].mean():.1f}')
    stat, p = stats.mannwhitneyu(t0['edad'].dropna(), t1['edad'].dropna(),
                                alternative='two-sided')
    print(f'Mann-Whitney: U={stat:.0f}, p={p:.4f}  '
        f'{"-> diferencia significativa" if p < 0.05 else "-> no significativa"}')



def target_vs_var_demo_dis(cols_demo, train):
    fig, axes = plt.subplots(1, len(cols_demo), figsize=(5 * len(cols_demo), 4))
    if len(cols_demo) == 1: axes = [axes]

    for ax, col in zip(axes, cols_demo):
        tasas = tasa_estres_cat(train, col)
        colores = [C1 if v >= 20 else C0 for v in tasas.values]
        ax.bar(tasas.index.astype(str), tasas.values, color=colores, alpha=0.85)
        ax.set_title(f'Tasa estrés por {col}', fontweight='bold')
        ax.set_ylabel('% estrés alto')
        ax.tick_params(axis='x', rotation=20)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))

        tabla_contingencia = pd.crosstab(train['estres_financiero_alto'], train[col])
        stat, p, _, _ = stats.chi2_contingency(tabla_contingencia)
        print(f'Chi-cuadrado con {col}: U={stat:.0f}, p={p}  '
            f'{"-> diferencia significativa" if p < 0.05 else "-> no significativa"}')
        
    plt.tight_layout()
    plt.savefig('src/img/eda_demograficas.png', bbox_inches='tight')
    plt.show()


def target_vs_var_lab(cols_lab, train):
    fig, axes = plt.subplots(1, len(cols_lab), figsize=(5 * len(cols_lab), 4))
    if len(cols_lab) == 1: axes = [axes]

    for ax, col in zip(axes, cols_lab):
        tasas = tasa_estres_cat(train, col)
        colores = [C1 if v >= 20 else C0 for v in tasas.values]
        ax.bar(tasas.index.astype(str), tasas.values, color=colores, alpha=0.85)
        ax.set_title(f'Tasa estrés por {col}', fontweight='bold')
        ax.set_ylabel('% estrés alto')
        ax.tick_params(axis='x', rotation=20)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))

        tabla_contingencia = pd.crosstab(train['estres_financiero_alto'], train[col])
        stat, p, _, _ = stats.chi2_contingency(tabla_contingencia)
        print(f'Chi-cuadrado con {col}: U={stat:.0f}, p={p}  '
            f'{"-> diferencia significativa" if p < 0.05 else "-> no significativa"}')

    plt.tight_layout()
    plt.savefig('src/img/eda_contrato_jornada.png', bbox_inches='tight')
    plt.show()



def target_vs_hours_exp(train, T0, T1):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, titulo in zip(axes,
        ['horas_semana', 'anios_experiencia'],
        ['Horas trabajadas por semana', 'Años de experiencia laboral']):
        if col not in train.columns: continue
        ax.hist(T0[col].dropna(), bins=30, color=C0, alpha=0.6,
                label='Sin estrés', density=True)
        ax.hist(T1[col].dropna(), bins=30, color=C1, alpha=0.6,
                label='Estrés alto', density=True)
        m0 = T0[col].mean()
        m1 = T1[col].mean()
        ax.axvline(m0, color=C0, linestyle='--', lw=1.5, label=f'Media 0: {m0:.1f}')
        ax.axvline(m1, color=C1, linestyle='--', lw=1.5, label=f'Media 1: {m1:.1f}')
        ax.set_title(titulo, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Densidad')
        ax.legend(fontsize=8)

        stat, p = stats.mannwhitneyu(T0[col].dropna(), T1[col].dropna(),
                                alternative='two-sided')
        print(f'Mann-Whitney para {col}: U={stat:.0f}, p={p:.4f}  '
            f'{"-> diferencia significativa" if p < 0.05 else "-> no significativa"}')

    plt.tight_layout()
    plt.savefig('src/img/eda_horas_experiencia.png', bbox_inches='tight')
    plt.show()
