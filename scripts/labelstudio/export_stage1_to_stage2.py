"""
Convert stage-1 dossier annotations into Label Studio import tasks for stage 2.

Input
-----
A JSON file containing an array of stage-1 annotation dicts (our internal format):
    [{"dossier_id": ..., "text": ..., "emails": [{"email_id": ...,
      "start_char": ..., "end_char": ...}]}, ...]

This can be:
  - Synthetic data written by make_synthetic_data.py (stage1/train.json)
  - Stage-1 Label Studio exports converted by convert_labelstudio_export.py --stage 1

Output
------
A directory of Label Studio import task files, one JSON per dossier:
    <output_dir>/<dossier_id>.json

Each file is a JSON array of tasks — one task per email in that dossier.
Each task retains (dossier_id, email_id) for provenance tracing.

Usage
-----
    python scripts/labelstudio/export_stage1_to_stage2.py \\
        --input data/synthetic/stage1/dev.json \\
        --output_dir data/labelstudio/stage2_import/

    # Then in Label Studio: Projects → Import → upload each JSON file.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def stage1_annotation_to_tasks(annotation: dict) -> list[dict]:
    """
    Convert one stage-1 dossier annotation to a list of Label Studio import tasks.

    Each task corresponds to one detected email within the dossier.
    The email text is sliced from the dossier text using the annotated char spans.

    Parameters
    ----------
    annotation : stage-1 annotation dict with keys dossier_id, text, emails

    Returns
    -------
    List of Label Studio task dicts ready for import.
    """
    dossier_id = annotation["dossier_id"]
    dossier_text = annotation["text"]
    tasks: list[dict] = []

    for email in annotation["emails"]:
        email_text = dossier_text[email["start_char"] : email["end_char"]]
        tasks.append(
            {
                "data": {
                    "text": email_text,
                    "dossier_id": dossier_id,
                    "email_id": email["email_id"],
                }
            }
        )

    return tasks


def convert(input_path: Path, output_dir: Path) -> None:
    """
    Process all dossiers in input_path and write one task file per dossier.
    """
    with open(input_path, encoding="utf-8") as f:
        stage1_data: list[dict] = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    total_tasks = 0

    for annotation in stage1_data:
        dossier_id = annotation["dossier_id"]
        tasks = stage1_annotation_to_tasks(annotation)
        out_path = output_dir / f"{dossier_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)
        total_tasks += len(tasks)
        print(f"  {dossier_id}: {len(tasks)} task(s) → {out_path}")

    print(f"\nTotal: {len(stage1_data)} dossiers → {total_tasks} stage-2 tasks in {output_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert stage-1 dossier annotations to Label Studio stage-2 import tasks."
    )
    parser.add_argument("--input", required=True, help="Path to stage-1 annotation JSON")
    parser.add_argument("--output_dir", required=True, help="Directory to write task files into")
    args = parser.parse_args(argv)

    convert(Path(args.input), Path(args.output_dir))


if __name__ == "__main__":
    main()
