import seaborn as sns

from pathlib import Path

PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
sns.set_palette(PALETTE)

# Raíz del proyecto (src/utils/ → src/ → project root)
_ROOT = Path(__file__).resolve().parent.parent.parent

# Rutas absolutas ancladas a la ubicación del archivo
DATA_DIR_BRONZE  = _ROOT / 'src/data/01_bronze'
DATA_DIR_SILVER  = _ROOT / 'src/data/02_silver'
INFO_DIR  = DATA_DIR_BRONZE / 'info'
PATH_OUT  = DATA_DIR_SILVER / 'dataset_analitico.csv'