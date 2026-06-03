"""
Prepare a Label Studio import task from a Woo dossier PDF.

Extracts the text and writes a JSON file that can be uploaded directly to
Label Studio. No trained model needed — this is the first step before
annotation.

Usage
-----
    # One dossier
    python scripts/prepare_import.py data/raw/21390238.pdf

    # All dossiers at once
    python scripts/prepare_import.py data/raw/*.pdf

    # Custom output directory
    python scripts/prepare_import.py data/raw/21390238.pdf --output_dir data/import/

Output
------
One JSON file per PDF, written to data/import/ by default.
Upload these files to Label Studio: Import → select file.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from pdf_extract import extract_text


def prepare(pdf_path: Path, output_dir: Path) -> None:
    dossier_id = pdf_path.stem  # filename without .pdf

    print(f"Extracting {pdf_path.name} …", end=" ", flush=True)
    text, page_map = extract_text(pdf_path)
    print(f"{len(page_map)} pages, {len(text):,} chars")

    task = [{"data": {"text": text, "dossier_id": dossier_id}}]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{dossier_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(task, f, ensure_ascii=False, indent=2)

    print(f"  → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Label Studio import tasks from Woo dossier PDFs."
    )
    parser.add_argument("pdfs", nargs="+", help="PDF file(s) to process")
    parser.add_argument(
        "--output_dir",
        default=str(_ROOT / "data" / "import"),
        help="Directory to write import JSON files (default: data/import/)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    for pdf in args.pdfs:
        prepare(Path(pdf), output_dir)

    print(f"\nDone. Import the JSON file(s) from {output_dir} into Label Studio.")


if __name__ == "__main__":
    main()
