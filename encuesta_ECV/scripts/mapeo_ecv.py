"""
Script de mapeo automatizado para ficheros de la Encuesta de Condiciones de Vida (ECV).

Empareja automáticamente los CSVs de datos con sus XLSXs de info por número de prefijo
(01_datos → 01_info, 02_datos → 02_info, etc.) dentro de la misma carpeta.

Uso:
    python mapeo_ecv.py <carpeta_datos> <carpeta_info> [carpeta_salida]

    # Si los datos y la info están en la misma carpeta:
    python mapeo_ecv.py data/ECV_2025/ data/ECV_2025/info/

    # Con carpeta de salida explícita:
    python mapeo_ecv.py data/ECV_2025/ data/ECV_2025/info/ resultados/

Genera por cada par:
  - <nombre>_mapeado.csv  →  datos con columnas renombradas + columnas _DESC de etiquetas
"""

import re
import sys
import pandas as pd
import numpy as np
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# 1. EMPAREJAR ARCHIVOS POR PREFIJO NUMÉRICO
# ──────────────────────────────────────────────────────────────────────────────

def emparejar_archivos(carpeta_datos: Path, carpeta_info: Path) -> list[tuple[Path, Path]]:
    """
    Devuelve lista de (csv_path, xlsx_path) emparejados por su prefijo numérico.
    Ej: 01_datos_hogar.csv  ↔  01_info_datos_hogar.xlsx
    """
    csvs  = {re.match(r'^(\d+)', f.stem).group(1): f
             for f in carpeta_datos.glob("*.csv")
             if re.match(r'^\d+', f.stem)}
    xlsxs = {re.match(r'^(\d+)', f.stem).group(1): f
             for f in carpeta_info.glob("*.xlsx")
             if re.match(r'^\d+', f.stem)}

    pares = []
    for num in sorted(set(csvs) & set(xlsxs)):
        pares.append((csvs[num], xlsxs[num]))
        print(f"  [{num}] {csvs[num].name}  ↔  {xlsxs[num].name}")

    sin_par_csv  = set(csvs) - set(xlsxs)
    sin_par_xlsx = set(xlsxs) - set(csvs)
    if sin_par_csv:
        print(f"  ⚠ CSVs sin info: {[csvs[n].name for n in sin_par_csv]}")
    if sin_par_xlsx:
        print(f"  ⚠ XLSXs sin datos: {[xlsxs[n].name for n in sin_par_xlsx]}")

    return pares


# ──────────────────────────────────────────────────────────────────────────────
# 2. LEER HOJA DISEÑO → {variable: descripcion}
# ──────────────────────────────────────────────────────────────────────────────

def leer_disenio(xl_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """
    Devuelve dos dicts:
      - desc_map:  { 'VAR': 'descripcion legible' }
      - tabla_map: { 'VAR': 'NombreTabla' }   (de la col "Diccionario de la variable")
        junto con la hoja donde buscar: tabla_hoja_map { 'VAR': 'Tablas2' }
    """
    raw = pd.read_excel(xl_path, sheet_name="Diseño", header=None)

    # Fila 1 siempre contiene los encabezados
    headers = [str(v).strip() for v in raw.iloc[1]]
    df = raw.iloc[2:].copy()
    df.columns = headers
    df = df.reset_index(drop=True)

    # Identificar columnas relevantes de forma flexible
    col_var   = next(c for c in df.columns if re.search(r'^variable', c, re.I))
    col_desc  = next(c for c in df.columns if re.search(r'descripci', c, re.I))
    col_dic   = next(c for c in df.columns if re.search(r'diccionario de la variable', c, re.I))
    col_hoja  = next(c for c in df.columns if re.search(r'hoja|ubic', c, re.I))

    desc_map       = {}  # VAR → descripción
    tabla_map      = {}  # VAR → nombre de tabla (ej: "T_Flag", "TD040B")
    tabla_hoja_map = {}  # VAR → hoja donde está esa tabla (ej: "Tablas1")

    for _, row in df.iterrows():
        var = str(row[col_var]).strip() if pd.notna(row[col_var]) else ""
        # Filtrar filas que no son variables reales
        if not var or var.startswith("*") or var.startswith("(") or var == "nan":
            continue
        # Limpiar saltos de línea que a veces aparecen en el nombre de variable
        var = var.split("\n")[0].strip()

        desc = str(row[col_desc]).strip() if pd.notna(row[col_desc]) else ""
        desc = desc.split("\n")[0].strip()  # quedarse con la primera línea
        if desc and desc != "nan":
            desc_map[var.upper()] = desc

        dic_var = str(row[col_dic]).strip() if pd.notna(row[col_dic]) else ""
        hoja    = str(row[col_hoja]).strip() if pd.notna(row[col_hoja]) else ""
        if dic_var and dic_var != "nan":
            tabla_map[var.upper()]      = dic_var.split("\n")[0].strip()
            tabla_hoja_map[var.upper()] = hoja.split("\n")[0].strip()

    return desc_map, tabla_map, tabla_hoja_map


# ──────────────────────────────────────────────────────────────────────────────
# 3. LEER TODAS LAS HOJAS DE TABLAS → { nombre_tabla: {codigo: etiqueta} }
# ──────────────────────────────────────────────────────────────────────────────

def leer_tablas(xl_path: Path) -> dict[str, dict]:
    """
    Recorre todas las hojas de tipo Tablas* y construye:
        { 'T_Flag': {-2: 'No aplicable', -1: 'No consta', 1: 'Variable completada'},
          'TD040B': {'ES11': 'Galicia', ...}, ... }

    Cada bloque tiene forma:
        NombreTabla  |  NaN  |  "En hoja … Variables: VAR1 ***"
        Código       |  Descripción
        val1         |  etiqueta1
        ...
        (línea vacía = nuevo bloque)
    """
    xf = pd.ExcelFile(xl_path)
    hojas_tablas = [s for s in xf.sheet_names
                    if s not in ("Diseño", "Tbls2-Detalle") and "tabla" in s.lower()]

    tablas: dict[str, dict] = {}

    for hoja in hojas_tablas:
        raw = pd.read_excel(xl_path, sheet_name=hoja, header=None).reset_index(drop=True)

        i = 0
        while i < len(raw):
            row = raw.iloc[i]
            c0 = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
            c2 = str(row.iloc[2]).strip() if (len(row) > 2 and pd.notna(row.iloc[2])) else ""

            # Inicio de bloque: c0 es nombre de tabla, c2 tiene texto de referencia
            if c0 and c0.lower() not in ("nan", "código", "codigo") and c2:
                nombre_tabla = c0.split("\n")[0].strip()

                # Avanzar hasta la fila "Código / Descripción"
                i += 1
                while i < len(raw):
                    r0 = str(raw.iloc[i].iloc[0]).strip().lower()
                    if r0 in ("código", "codigo"):
                        i += 1
                        break
                    i += 1

                # Leer pares código-etiqueta hasta línea vacía
                codigos: dict = {}
                while i < len(raw):
                    r = raw.iloc[i]
                    cod = r.iloc[0]
                    etq = r.iloc[1] if len(r) > 1 else np.nan
                    if pd.isna(cod) and pd.isna(etq):
                        break
                    if pd.notna(cod):
                        cod_str = str(cod).strip().split("\n")[0]
                        try:
                            key = int(float(cod_str))
                        except (ValueError, TypeError):
                            key = cod_str
                        codigos[key] = str(etq).strip() if pd.notna(etq) else ""
                    i += 1

                if codigos:
                    tablas[nombre_tabla] = codigos
            else:
                i += 1

    return tablas


# ──────────────────────────────────────────────────────────────────────────────
# 4. CONSTRUIR MAPA VARIABLE → {codigo: etiqueta}
#    usando tabla_map (qué tabla le corresponde) + tablas (contenido de tablas)
# ──────────────────────────────────────────────────────────────────────────────

def construir_mapeo_valores(tabla_map: dict, tablas: dict) -> dict[str, dict]:
    """
    Para cada variable que tiene asignada una tabla en el Diseño,
    busca esa tabla en el dict de tablas y devuelve el mapeo directo
    { 'VAR': {codigo: etiqueta} }.
    """
    mapeo: dict[str, dict] = {}
    for var, nombre_tabla in tabla_map.items():
        nombre_limpio = nombre_tabla.split("\n")[0].strip()
        if nombre_limpio in tablas:
            mapeo[var] = tablas[nombre_limpio]
    return mapeo


# ──────────────────────────────────────────────────────────────────────────────
# 5. PROCESAR UN PAR CSV + XLSX
# ──────────────────────────────────────────────────────────────────────────────

def procesar_par(csv_path: Path, xl_path: Path, out_dir: Path) -> None:
    print(f"\n{'─'*60}")
    print(f"  CSV : {csv_path.name}")
    print(f"  INFO: {xl_path.name}")

    # Leer CSV
    df = None
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(csv_path, encoding=enc, low_memory=False)
            print(f"  Filas: {len(df):,} | Columnas: {len(df.columns)} | Enc: {enc}")
            break
        except UnicodeDecodeError:
            continue

    # Leer diseño y tablas
    desc_map, tabla_map, _ = leer_disenio(xl_path)
    tablas                  = leer_tablas(xl_path)
    valor_map               = construir_mapeo_valores(tabla_map, tablas)

    df_out = df.copy()

    # ── Insertar columnas _DESC con etiquetas de valores ──────────────────────
    n_desc = 0
    for col in df.columns:
        col_up = col.strip().upper()
        if col_up in valor_map:
            def mapper(v, vm=valor_map[col_up]):
                if pd.isna(v) or str(v).strip() in ("", "nan"):
                    return v
                try:
                    k = int(float(str(v).strip()))
                except (ValueError, TypeError):
                    k = str(v).strip()
                return vm.get(k, v)

            pos = df_out.columns.get_loc(col) + 1
            df_out.insert(pos, col + "_DESC", df[col].map(mapper))
            n_desc += 1

    # ── Renombrar columnas con descripción del Diseño ─────────────────────────
    rename  = {}
    seen    = {}
    for col in df_out.columns:
        base = col[:-5] if col.endswith("_DESC") else col
        desc = desc_map.get(base.strip().upper())
        if desc:
            nuevo = desc + (" (etiqueta)" if col.endswith("_DESC") else "")
        else:
            nuevo = col
        seen[nuevo] = seen.get(nuevo, 0) + 1
        if seen[nuevo] > 1:
            nuevo = f"{nuevo} ({seen[nuevo]})"
        rename[col] = nuevo

    df_out = df_out.rename(columns=rename)
    n_renombradas = sum(1 for k, v in rename.items() if k != v)
    print(f"  Columnas _DESC añadidas: {n_desc} | Columnas renombradas: {n_renombradas}")

    # ── Guardar ───────────────────────────────────────────────────────────────
    out_path = out_dir / (csv_path.stem + "_mapeado.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  ✓ Guardado: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. PUNTO DE ENTRADA
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    carpeta_datos = Path('data/ECV_2025/')
    carpeta_info  = Path('data/ECV_2025/info/')
    carpeta_salida = Path('data/ECV_2025/mapeo')
    # carpeta_salida.mkdir(parents=True, exist_ok=True)

    print(f"Datos : {carpeta_datos}")
    print(f"Info  : {carpeta_info}")
    print(f"Salida: {carpeta_salida}\n")
    print("Emparejando archivos...")
    pares = emparejar_archivos(carpeta_datos, carpeta_info)

    if not pares:
        print("No se encontraron pares. Verifica que los archivos empiecen con prefijo numérico (01_, 02_...).")
        sys.exit(1)

    for csv_p, xl_p in pares:
        procesar_par(csv_p, xl_p, carpeta_salida)

    print(f"\n{'═'*60}")
    print(f"Proceso completado. {len(pares)} fichero(s) procesado(s) en: {carpeta_salida}")
