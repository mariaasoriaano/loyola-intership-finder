import pandas as pd
from pathlib import Path


def excel_to_markdown_csv(excel_path: Path, output_dir: Path):
    """
    Convierte un Excel en Markdown y lo guarda dentro de un CSV
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / excel_path.with_suffix(".csv").name

    xls = pd.ExcelFile(excel_path)
    markdown_blocks = []

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)

        markdown_blocks.append(f"# {sheet_name}\n")

        if df.empty:
            markdown_blocks.append("_Hoja vacía_\n")
        else:
            markdown_blocks.append(df.to_markdown(index=False) + "\n")

    markdown_content = "\n".join(markdown_blocks)

    pd.DataFrame(
        {"markdown": [markdown_content]}
    ).to_csv(output_csv, index=False, encoding="utf-8")

    print(f"✔ Generado: {output_csv}")


def process_companies(companies_dir: str, output_dir: str):
    companies_path = Path(companies_dir)
    output_path = Path(output_dir)

    if not companies_path.exists():
        raise FileNotFoundError(f"No existe la carpeta: {companies_path}")

    for company_folder in companies_path.iterdir():
        if company_folder.is_dir():
            for excel_file in company_folder.glob("*.xlsx"):
                excel_to_markdown_csv(excel_file, output_path)


if __name__ == "__main__":
    COMPANIES_DIR = "companies"
    OUTPUT_DIR = "data"

    process_companies(COMPANIES_DIR, OUTPUT_DIR)