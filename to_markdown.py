import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# =========================================================
# Utilidades
# =========================================================

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def is_empty(x) -> bool:
    return x is None or (isinstance(x, float) and pd.isna(x)) or str(x).strip() == ""


def looks_like_date_ddmmyyyy(x) -> bool:
    if is_empty(x):
        return False
    s = str(x).strip()
    return bool(re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", s))


def parse_year_from_date(x) -> Optional[int]:
    if is_empty(x):
        return None
    s = str(x).strip()
    m = re.search(r"(\d{4})", s)
    return int(m.group(1)) if m else None


def to_float(x) -> Optional[float]:
    if is_empty(x):
        return None
    try:
        s = str(x).strip().replace(" ", "")
        if "," in s and "." in s:
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", ".")
        return float(s)
    except Exception:
        return None


def format_num(x) -> str:
    if is_empty(x):
        return "—"
    val = to_float(x)
    if val is None:
        return str(x).strip()
    if abs(val) >= 1000:
        return f"{val:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{val:.2f}".replace(".", ",")


def percent(curr: Optional[float], prev: Optional[float]) -> str:
    if curr is None or prev is None or prev == 0:
        return "—"
    return f"{((curr - prev) / abs(prev)) * 100:.1f}%"


def read_xls_report(path: Path) -> pd.DataFrame:
    """
    Lee el xls/xlsx como DF sin headers (porque vienen maquetados).
    """
    # En tus archivos existe la hoja "Page 1"
    return pd.read_excel(path, sheet_name="Page 1", header=None)


# =========================================================
# INFO.XLS -> Resumen
# =========================================================

# Labels que queremos capturar (normalizados) -> nombre bonito en Markdown
INFO_WANTED = {
    "código nif": "NIF",
    "telefono": "Teléfono",
    "teléfono": "Teléfono",
    "direccion web": "Web",
    "dirección web": "Web",
    "número de empleados": "Empleados",
    "numero de empleados": "Empleados",
    "director ejecutivo": "Directivo",
}

# Otros campos que podemos sacar si existen
OPTIONAL_FIELDS = {
    "forma jurídica": "Forma jurídica",
    "fecha constitución": "Fecha constitución",
    "capital social (eur)": "Capital social (EUR)",
    "estado": "Estado",
}

LABEL_BLOCKLIST = set([norm(k) for k in list(INFO_WANTED.keys()) + list(OPTIONAL_FIELDS.keys())])


def extract_company_name_and_location(df_info: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    En tus Info:
      - Nombre suele estar en (fila 0, col 1)
      - Localización suele estar en (fila 2, col 0)
    """
    name = None
    loc = None

    try:
        v = df_info.iat[0, 1]
        if not is_empty(v):
            name = str(v).strip()
    except Exception:
        pass

    try:
        v = df_info.iat[2, 0]
        if not is_empty(v):
            loc = str(v).strip()
    except Exception:
        pass

    return name, loc


def nearest_value_in_row(row: List[object], label_col: int, max_scan: int = 40) -> Optional[str]:
    """
    Dado un label en una columna, busca el valor asociado en la misma fila.
    FIX CLAVE: primero busca a la DERECHA (que es como vienen Teléfono/Web/NIF),
    si no encuentra, busca a la IZQUIERDA.

    Devuelve el valor no vacío más cercano evitando coger otra etiqueta.
    """
    # 1) derecha
    for j in range(label_col + 1, min(len(row), label_col + 1 + max_scan)):
        v = row[j]
        if is_empty(v):
            continue
        # evita que el valor sea otra etiqueta
        if norm(v) in LABEL_BLOCKLIST:
            continue
        return str(v).strip()

    # 2) izquierda
    for j in range(label_col - 1, max(-1, label_col - 1 - max_scan), -1):
        v = row[j]
        if is_empty(v):
            continue
        if norm(v) in LABEL_BLOCKLIST:
            continue
        return str(v).strip()

    return None


def extract_labels(df: pd.DataFrame, wanted: Dict[str, str]) -> Dict[str, str]:
    """
    Recorre todas las celdas buscando labels (wanted keys) y extrae el valor asociado.
    """
    found: Dict[str, str] = {}

    for i in range(df.shape[0]):
        row = df.iloc[i, :].tolist()
        for j in range(len(row)):
            cell = row[j]
            if is_empty(cell):
                continue

            cell_n = norm(cell)
            for label in wanted.keys():
                lab_n = norm(label)
                if cell_n == lab_n or lab_n in cell_n:
                    val = nearest_value_in_row(row, j)
                    if val:
                        found[lab_n] = val
    return found


def extract_description_activity(df: pd.DataFrame) -> Optional[str]:
    """
    Busca la sección "Descripción actividad" y devuelve la primera línea de descripción real.
    En tus archivos suele estar justo debajo en la columna 0.
    """
    target = "descripción actividad"
    for i in range(df.shape[0]):
        v = df.iat[i, 0]
        if is_empty(v):
            continue
        if norm(v) == norm(target):
            # buscamos hacia abajo el primer texto "normal" en col 0
            for k in range(i + 1, min(i + 20, df.shape[0])):
                desc = df.iat[k, 0]
                if is_empty(desc):
                    continue
                # corta si aparece otro "título"
                if "código" in norm(desc) or "cnae" in norm(desc):
                    break
                return str(desc).strip()
    return None


def summarize_info(info_path: Path) -> str:
    df = read_xls_report(info_path)

    company_name, location = extract_company_name_and_location(df)

    # Extraemos campos clave (web/tel/nif/empleados/directivo)
    found_main = extract_labels(df, INFO_WANTED)
    found_optional = extract_labels(df, OPTIONAL_FIELDS)

    description = extract_description_activity(df)

    md: List[str] = []
    md.append("## Resumen (Info)\n")

    if company_name:
        md.append(f"- **Nombre**: {company_name}")
    if location:
        md.append(f"- **Ubicación**: {location}")

    if description:
        md.append(f"- **Descripción**: {description}")

    # Orden lógico
    order = ["código nif", "dirección web", "direccion web", "teléfono", "telefono", "número de empleados", "numero de empleados", "director ejecutivo"]
    for key in order:
        k = norm(key)
        if k in found_main:
            md.append(f"- **{INFO_WANTED[k]}**: {found_main[k]}")

    # opcionales si existen
    opt_order = ["forma jurídica", "fecha constitución", "capital social (eur)", "estado"]
    for key in opt_order:
        k = norm(key)
        if k in found_optional:
            md.append(f"- **{OPTIONAL_FIELDS[k]}**: {found_optional[k]}")

    md.append("")
    return "\n".join(md)


# =========================================================
# PROFIT.XLS -> Resumen (igual que antes)
# =========================================================

KPI_ALIASES = {
    "ventas": ["importe neto cifra de ventas", "cifra de ventas", "ventas"],
    "ingresos_explotacion": ["ingresos de explotación", "ingresos explotación"],
    "resultado_explotacion": ["resultado explotación", "resultado de explotación", "resultado explotacion", "ebit"],
    "resultado_ejercicio": ["resultado del ejercicio", "resultado ejercicio"],
    "ebitda": ["ebitda"],
    "total_activo": ["total activo"],
    "fondos_propios": ["fondos propios", "patrimonio neto", "equity"],
    "pasivo_fijo": ["pasivo fijo"],
    "pasivo_circulante": ["pasivo circulante"],
    "tesoreria": ["tesorería", "tesoreria"],
    "deudores": ["deudores"],
    "existencias": ["existencias"],
    "empleados": ["número empleados", "numero empleados", "empleados"],
}


def detect_header_years(df_profit: pd.DataFrame) -> Tuple[int, List[Tuple[int, int]]]:
    best_row = None
    best_hits: List[Tuple[int, int]] = []

    for i in range(min(30, df_profit.shape[0])):
        row = df_profit.iloc[i, :].tolist()
        hits = []
        for j, cell in enumerate(row):
            if looks_like_date_ddmmyyyy(cell):
                y = parse_year_from_date(cell)
                if y:
                    hits.append((j, y))
        if len(hits) > len(best_hits):
            best_hits = hits
            best_row = i

    if best_row is None or not best_hits:
        return 0, []
    return best_row, best_hits


def score_numeric_column(df: pd.DataFrame, col: int, start_row: int) -> int:
    score = 0
    for i in range(start_row, min(start_row + 60, df.shape[0])):
        v = df.iat[i, col]
        if to_float(v) is not None:
            score += 1
    return score


def map_year_to_value_column(df_profit: pd.DataFrame, header_row: int, date_cols_years: List[Tuple[int, int]]) -> Dict[int, int]:
    year_to_col: Dict[int, int] = {}
    start_measure = header_row + 10

    for date_col, year in date_cols_years:
        best_col = None
        best_score = -1

        for c in range(max(0, date_col - 8), date_col):
            sc = score_numeric_column(df_profit, c, start_measure)
            if sc > best_score:
                best_score = sc
                best_col = c

        if best_col is not None and best_score > 0:
            year_to_col[year] = best_col

    return year_to_col


def find_kpi_row(df_profit: pd.DataFrame, aliases: List[str]) -> Optional[int]:
    aliases_n = [norm(a) for a in aliases]
    for i in range(df_profit.shape[0]):
        v = df_profit.iat[i, 0]
        if is_empty(v):
            continue
        txt = norm(v)
        if any(a in txt for a in aliases_n):
            return i
    return None


def extract_kpis_last_years(df_profit: pd.DataFrame, year_to_col: Dict[int, int], years: List[int]) -> Dict[str, Dict[int, object]]:
    out: Dict[str, Dict[int, object]] = {}

    for kpi_name, aliases in KPI_ALIASES.items():
        row_idx = find_kpi_row(df_profit, aliases)
        if row_idx is None:
            continue
        out[kpi_name] = {}
        for y in years:
            if y in year_to_col:
                col = year_to_col[y]
                out[kpi_name][y] = df_profit.iat[row_idx, col]
    return out


def summarize_profit(profit_path: Path) -> str:
    df = read_xls_report(profit_path)

    header_row, date_cols_years = detect_header_years(df)
    if not date_cols_years:
        return "## Resumen (Profit)\n_No se detectó cabecera de años (fechas 31/12/AAAA)._ \n"

    year_to_col = map_year_to_value_column(df, header_row, date_cols_years)
    years_sorted = sorted(year_to_col.keys(), reverse=True)
    years = years_sorted[:3]

    kpis = extract_kpis_last_years(df, year_to_col, years)

    md: List[str] = []
    md.append("## Resumen (Profit)\n")

    rows = []
    core_order = [
        ("Ventas", "ventas"),
        ("Ingresos explotación", "ingresos_explotacion"),
        ("EBITDA", "ebitda"),
        ("Resultado explotación (EBIT)", "resultado_explotacion"),
        ("Resultado del ejercicio", "resultado_ejercicio"),
        ("Total activo", "total_activo"),
        ("Fondos propios", "fondos_propios"),
        ("Pasivo fijo", "pasivo_fijo"),
        ("Tesorería", "tesoreria"),
        ("Empleados", "empleados"),
    ]

    for label, key in core_order:
        if key not in kpis:
            continue
        r = {"KPI": label}
        for y in years:
            r[str(y)] = format_num(kpis[key].get(y))
        rows.append(r)

    if rows:
        df_out = pd.DataFrame(rows)
        md.append("### KPIs (últimos años)\n")
        md.append(df_out.to_markdown(index=False))
        md.append("")
    else:
        md.append("_No se pudieron extraer KPIs del Profit con los patrones actuales._\n")

    md.append("### Highlights\n")

    def get_val(k: str, y: int) -> Optional[float]:
        if k not in kpis or y not in kpis[k]:
            return None
        return to_float(kpis[k][y])

    if len(years) >= 2:
        y0, y1 = years[0], years[1]
        ventas_y0 = get_val("ventas", y0)
        ventas_y1 = get_val("ventas", y1)
        if ventas_y0 is not None and ventas_y1 is not None:
            md.append(f"- **Crecimiento ventas {y0} vs {y1}**: {percent(ventas_y0, ventas_y1)}")

        ebitda_y0 = get_val("ebitda", y0)
        if ebitda_y0 is not None and ventas_y0 not in (None, 0):
            md.append(f"- **Margen EBITDA {y0}**: {(ebitda_y0 / ventas_y0) * 100:.1f}%")

        res_y0 = get_val("resultado_ejercicio", y0)
        if res_y0 is not None and ventas_y0 not in (None, 0):
            md.append(f"- **Margen neto {y0}**: {(res_y0 / ventas_y0) * 100:.1f}%")

    md.append("")
    md.append(f"_Años detectados en Profit: {', '.join(map(str, sorted(year_to_col.keys(), reverse=True)[:10]))}_\n")
    return "\n".join(md)


# =========================================================
# Orquestación por empresa
# =========================================================

def process_companies(companies_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    if not companies_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta: {companies_dir}")

    for company_folder in companies_dir.iterdir():
        if not company_folder.is_dir():
            continue

        company_name = company_folder.name

        excel_files = [
            f for f in company_folder.rglob("*")
            if f.suffix.lower() in [".xlsx", ".xls", ".xlsm"]
        ]

        if not excel_files:
            print(f"⚠ Sin Excel en {company_name}")
            continue

        info_file = next((f for f in excel_files if "info" in norm(f.stem)), None)
        profit_file = next((f for f in excel_files if "profit" in norm(f.stem)), None)

        md_parts: List[str] = [f"# {company_name}\n"]

        if info_file:
            md_parts.append(summarize_info(info_file))
        else:
            md_parts.append("## Resumen (Info)\n_No encontrado archivo `Info`._\n")

        if profit_file:
            md_parts.append(summarize_profit(profit_file))
        else:
            md_parts.append("## Resumen (Profit)\n_No encontrado archivo `Profit`._\n")

        md_parts.append("## Archivos detectados\n")
        for f in sorted(excel_files):
            md_parts.append(f"- {f.name}")

        output_md = output_dir / f"{company_name}.md"
        output_md.write_text("\n".join(md_parts), encoding="utf-8")
        print(f"Generado: {output_md}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    COMPANIES_DIR = BASE_DIR / "companies"
    OUTPUT_DIR = BASE_DIR / "data"
    process_companies(COMPANIES_DIR, OUTPUT_DIR)
