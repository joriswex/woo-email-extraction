"""
Dossier management script.

Reads data/raw/dossiers.csv, matches each row to its PDF in data/raw/,
renames the PDF to an anonymous ID-based filename, and writes a
metadata.json alongside it.

Anonymous filename format
-------------------------
    {government_id}.pdf

Examples
    nl.oorg10004.2i.2024.9.pdf

The original filename (which may contain topic keywords) is stored only
inside metadata.json and never appears in the filesystem after renaming.
This keeps the data/raw/ folder anonymous by default.

CSV format (data/raw/dossiers.csv)
-----------------------------------
    original_filename  : exact filename as it currently sits in data/raw/
    government_id      : official government PID (e.g. nl.oorg10004.2i.2024.9)
    ministry           : short ministry code (e.g. minfin, bzk, rws)
    publish_date       : ISO date the dossier was published (YYYY-MM-DD)
    title              : short descriptive title (stored in metadata only)

Usage
-----
    python scripts/manage_dossiers.py
    python scripts/manage_dossiers.py --dry-run   # preview without renaming
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_RAW_DIR = _ROOT / "data" / "raw"
_CSV_PATH = _RAW_DIR / "dossiers.csv"


def load_csv() -> list[dict]:
    with open(_CSV_PATH, encoding="utf-8") as f:
        lines = [l for l in f if not l.startswith("#")]
    reader = csv.DictReader(lines)
    return [row for row in reader if row["original_filename"].strip()]


def process(dry_run: bool = False) -> None:
    rows = load_csv()
    if not rows:
        print("No rows found in dossiers.csv.")
        return

    ok = skipped = 0

    for row in rows:
        original_name = row["original_filename"].strip()
        government_id = row["government_id"].strip()
        ministry      = row["ministry"].strip()
        publish_date  = row["publish_date"].strip()
        title         = row["title"].strip()

        original_path = _RAW_DIR / original_name
        anonymous_name = f"{government_id}.pdf"
        anonymous_path = _RAW_DIR / anonymous_name
        metadata_path  = _RAW_DIR / f"{government_id}.json"

        # Skip if original file not found
        if not original_path.exists():
            # Maybe it was already renamed
            if anonymous_path.exists():
                print(f"  [already renamed]  {anonymous_name}")
                ok += 1
            else:
                print(f"  [NOT FOUND]        {original_name}")
                skipped += 1
            continue

        metadata = {
            "government_id":    government_id,
            "ministry":         ministry,
            "publish_date":     publish_date,
            "title":            title,
            "filename":         anonymous_name,
            "original_filename": original_name,
        }

        if dry_run:
            print(f"  [dry-run] would rename: {original_name}")
            print(f"                      to: {anonymous_name}")
            print(f"            metadata at: {metadata_path.name}")
            print()
        else:
            original_path.rename(anonymous_path)
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            print(f"  [renamed]  {original_name}")
            print(f"          →  {anonymous_name}")
            print(f"             {metadata_path.name} written")
            print()

        ok += 1

    print(f"Done: {ok} processed, {skipped} skipped (file not found).")
    if skipped:
        print("Check that original_filename in dossiers.csv matches exactly "
              "(including spaces and capitalisation).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename Woo dossier PDFs to anonymous IDs and write metadata."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would happen without renaming anything.",
    )
    args = parser.parse_args()
    process(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
