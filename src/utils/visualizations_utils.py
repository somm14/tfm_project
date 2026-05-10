import matplotlib.pyplot as plt

from matplotlib.patches import FancyBboxPatch

from scripts.constants_var import PALETTE


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
    plt.savefig('./img/bronze_diagrama_ficheros.png', bbox_inches='tight')
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
    plt.savefig('./img/silver_data_urbanizacion.png', bbox_inches='tight')
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
    plt.savefig('./img/silver_filtros_laborales.png', bbox_inches='tight')
    plt.show()