import pandas as pd
from pathlib import Path


def excel_to_document_markdown(excel_path: Path) -> str:
    xls = pd.ExcelFile(excel_path)
    blocks = [f"## Archivo: {excel_path.name}\n"]

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        blocks.append(f"### Hoja: {sheet_name}\n")

        for _, row in df.iterrows():
            values = [str(v).strip() for v in row if pd.notna(v)]

            if not values:
                blocks.append("")  # separación semántica
                continue

            # 1 celda → título o texto libre
            if len(values) == 1:
                blocks.append(f"**{values[0]}**")

            # 2 celdas → clave: valor
            elif len(values) == 2:
                blocks.append(f"- {values[0]}: {values[1]}")

            # más de 2 → texto concatenado
            else:
                blocks.append(" ".join(values))

        blocks.append("")

    return "\n".join(blocks)



def process_companies_text(companies_dir: Path, output_dir: Path):
    """
    Procesa todas las empresas y genera Markdown en texto plano por empresa
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not companies_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta: {companies_dir}")

    for company_folder in companies_dir.iterdir():
        if not company_folder.is_dir():
            continue

        company_name = company_folder.name
        markdown_company = [f"# Empresa: {company_name}\n"]

        excel_files = [
            f for f in company_folder.rglob("*")
            if f.suffix.lower() in [".xlsx", ".xls", ".xlsm"]
        ]

        if not excel_files:
            print(f"⚠ Sin Excel en {company_name}")
            continue

        for excel_file in excel_files:
            markdown_company.append(excel_to_document_markdown(excel_file))

        output_md = output_dir / f"{company_name}.md"

        with open(output_md, "w", encoding="utf-8") as f:
            f.write("\n".join(markdown_company))

        print(f"✔ Generado: {output_md}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent

    COMPANIES_DIR = BASE_DIR / "companies"
    OUTPUT_DIR = BASE_DIR / "data"

    process_companies_text(COMPANIES_DIR, OUTPUT_DIR)