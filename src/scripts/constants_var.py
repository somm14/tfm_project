import seaborn as sns

from pathlib import Path

PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
sns.set_palette(PALETTE)

# Rutas 
DATA_DIR_BRONZE  = Path('data/01_bronze/')
INFO_DIR  = DATA_DIR_BRONZE / 'info'
PATH_OUT  = DATA_DIR_BRONZE / 'dataset_analitico.csv'

# Códigos de valor perdido del INE
NULOS_INE = [-1, -2, -4, -5, -6]
