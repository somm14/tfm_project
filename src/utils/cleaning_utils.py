import pandas as pd

from scripts.constants_var import DATA_DIR_BRONZE

def cargar_csv(nombre: str) -> pd.DataFrame:
    '''Carga un CSV probando encodings comunes del INE.'''
    path = DATA_DIR_BRONZE / nombre
    for enc in ('utf-8', 'latin-1', 'cp1252'):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f'No se pudo leer {path}')