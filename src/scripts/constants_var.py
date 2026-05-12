import seaborn as sns

from pathlib import Path

PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
sns.set_palette(PALETTE)

# Rutas 
DATA_DIR_BRONZE  = Path('data/01_bronze/')
DATA_DIR_SILVER  = Path('data/02_silver/')
INFO_DIR  = DATA_DIR_BRONZE / 'info'
PATH_OUT  = DATA_DIR_SILVER / 'dataset_analitico.csv'