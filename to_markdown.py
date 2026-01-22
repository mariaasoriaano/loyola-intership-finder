import pandas as pd
from pathlib import Path


def excel_to_markdown(excel_path: Path) -> str:
    xls = pd.ExcelFile(excel_path)
    markdown_blocks = []

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)

        markdown_blocks.append(f"## {excel_path.name} - {sheet_name}\n")

        if df.empty:
            markdown_blocks.append("_Hoja vacía_\n")
        else:
            markdown_blocks.append(df.to_markdown(index=False) + "\n")

    return "\n".join(markdown_blocks)


def process_companies(companies_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    if not companies_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta: {companies_dir}")

    for company_folder in companies_dir.iterdir():
        if not company_folder.is_dir():
            continue

        company_name = company_folder.name
        markdown_company = [f"# {company_name}\n"]

        excel_files = [
            f for f in company_folder.rglob("*")
            if f.suffix.lower() in [".xlsx", ".xls", ".xlsm"]
]

        if not excel_files:
            print(f"⚠ Sin Excel en {company_name}")
            continue

        for excel_file in excel_files:
            markdown_company.append(excel_to_markdown(excel_file))

        output_csv = output_dir / f"{company_name}.csv"

        pd.DataFrame(
            {"markdown": ["\n".join(markdown_company)]}
        ).to_csv(output_csv, index=False, encoding="utf-8")

        print(f"✔ Generado: {output_csv}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent

    COMPANIES_DIR = BASE_DIR / "companies"
    OUTPUT_DIR = BASE_DIR / "data"

    process_companies(COMPANIES_DIR, OUTPUT_DIR)