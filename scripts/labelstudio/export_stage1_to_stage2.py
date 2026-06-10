"""
Convert stage-1 dossier annotations into Label Studio import tasks for stage 2.

Pre-annotations for field values (FROM, TO, CC, DATE, SUBJECT, ATTACHMENT)
are loaded from the original preannotate.py import files and
converted from dossier-level to email-relative offsets.

Usage
-----
    python scripts/labelstudio/export_stage1_to_stage2.py \\
        --input      data/annotations/stage1.json \\
        --output_dir data/labelstudio/stage2_import/

    # Optional: point to a different pre-annotation directory
    python scripts/labelstudio/export_stage1_to_stage2.py \\
        --input      data/annotations/stage1.json \\
        --output_dir data/labelstudio/stage2_import/ \\
        --import_dir data/import/
"""

from __future__ import annotations
import argparse
import json
import uuid
from pathlib import Path

_ROOT        = Path(__file__).resolve().parent.parent.parent
_IMPORT_DIR  = _ROOT / "data" / "import"
_OUTPUT_DIR  = _ROOT / "data" / "import" / "stage2"

FIELD_LABELS = {"FROM", "TO", "CC", "DATE", "SUBJECT", "ATTACHMENT"}


def _load_preannotations(import_dir: Path, dossier_id: str) -> list[tuple[int, int, str]]:
    """Return (start, end, label) field spans from the preannotate import file."""
    p = import_dir / f"{dossier_id}.json"
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    spans = []
    for pred in data[0].get("predictions", []):
        for r in pred.get("result", []):
            labels = r.get("value", {}).get("labels", [])
            if labels and labels[0] in FIELD_LABELS:
                spans.append((r["value"]["start"], r["value"]["end"], labels[0]))
    return spans


def stage1_annotation_to_tasks(
    annotation: dict,
    preannotations: list[tuple[int, int, str]],
) -> list[dict]:
    dossier_id   = annotation["dossier_id"]
    dossier_text = annotation["text"]
    tasks: list[dict] = []

    for email in annotation["emails"]:
        e_start    = email["start_char"]
        e_end      = email["end_char"]
        email_text = dossier_text[e_start:e_end]

        # Convert dossier-level field spans to email-relative offsets
        results = []
        for f_start, f_end, label in preannotations:
            if f_start >= e_start and f_end <= e_end:
                results.append({
                    "id": str(uuid.uuid4())[:8],
                    "type": "labels",
                    "from_name": "label",
                    "to_name": "text",
                    "value": {
                        "start":  f_start - e_start,
                        "end":    f_end   - e_start,
                        "labels": [label],
                    },
                })

        task: dict = {
            "data": {
                "text":       email_text,
                "dossier_id": dossier_id,
                "email_id":   email["email_id"],
            }
        }
        if results:
            task["predictions"] = [{
                "model_version": "regex-v1",
                "score": 0.7,
                "result": results,
            }]

        tasks.append(task)

    return tasks


def convert(input_path: Path, output_dir: Path, import_dir: Path) -> None:
    with open(input_path, encoding="utf-8") as f:
        stage1_data: list[dict] = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    total_tasks = total_preannotated = 0

    for annotation in stage1_data:
        dossier_id    = annotation["dossier_id"]
        preannotations = _load_preannotations(import_dir, dossier_id)
        tasks         = stage1_annotation_to_tasks(annotation, preannotations)

        out_path = output_dir / f"{dossier_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)

        n_pre = sum(1 for t in tasks if "predictions" in t)
        total_tasks        += len(tasks)
        total_preannotated += n_pre
        src = "pre-annot." if preannotations else "no pre-annot."
        print(f"  {dossier_id}: {len(tasks)} emails  ({n_pre} with field pre-annotations)  [{src}]")

    print(f"\nTotal: {len(stage1_data)} dossiers → {total_tasks} tasks "
          f"({total_preannotated} with pre-annotations) in {output_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert stage-1 annotations to Label Studio stage-2 import tasks."
    )
    parser.add_argument("--input",      required=True, help="stage-1 annotation JSON")
    parser.add_argument(
        "--output_dir",
        default=str(_OUTPUT_DIR),
        help=f"directory to write stage-2 task files (default: {_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--import_dir",
        default=str(_IMPORT_DIR),
        help=f"directory with preannotate.py import files (default: {_IMPORT_DIR})",
    )
    args = parser.parse_args(argv)

    convert(Path(args.input), Path(args.output_dir), Path(args.import_dir))


if __name__ == "__main__":
    main()
