"""
Create blank Label Studio import tasks for inter-rater agreement (IRA).

Two dossiers are used for IRA across both stages:
  - 125051652  (115 emails, strong ATTACHMENT + CC coverage)
  - 17921044   (64 emails, diverse field types, manageable size)

Tasks contain NO pre-annotations so both raters annotate from scratch.

Usage
-----
    python scripts/ira/create_ira_tasks.py

Output
------
    data/ira/stage1_ira_tasks.json   — import into your Stage 1 Label Studio project
    data/ira/stage2_ira_tasks.json   — import into your Stage 2 Label Studio project
"""
from __future__ import annotations

import json
from pathlib import Path

_ROOT      = Path(__file__).resolve().parent.parent.parent
_ANN_DIR   = _ROOT / "data" / "annotations"
_IRA_DIR   = _ROOT / "data" / "ira"

IRA_DOSSIERS = ["125051652", "17921044"]


def create_stage1_tasks() -> None:
    stage1 = json.load(open(_ANN_DIR / "stage1.json", encoding="utf-8"))
    tasks = []
    for record in stage1:
        if record["dossier_id"] not in IRA_DOSSIERS:
            continue
        # Blank task — no pre-annotations, rater marks email spans from scratch
        tasks.append({
            "data": {
                "text":       record["text"],
                "dossier_id": record["dossier_id"],
            }
        })
        print(f"  Stage 1 — {record['dossier_id']}: {len(record['emails'])} emails "
              f"({len(record['text'])} chars)")

    out = _IRA_DIR / "stage1_ira_tasks.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
    print(f"  → {out}  ({len(tasks)} tasks)\n")


def create_stage2_tasks() -> None:
    stage2 = json.load(open(_ANN_DIR / "stage2.json", encoding="utf-8"))
    tasks = []
    for record in stage2:
        if record["dossier_id"] not in IRA_DOSSIERS:
            continue
        # Blank task — no pre-annotations, rater labels field spans from scratch
        tasks.append({
            "data": {
                "text":       record["text"],
                "dossier_id": record["dossier_id"],
                "email_id":   record["email_id"],
            }
        })

    by_dossier: dict[str, int] = {}
    for t in tasks:
        did = t["data"]["dossier_id"]
        by_dossier[did] = by_dossier.get(did, 0) + 1
    for did, n in sorted(by_dossier.items()):
        print(f"  Stage 2 — {did}: {n} emails")

    out = _IRA_DIR / "stage2_ira_tasks.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
    print(f"  → {out}  ({len(tasks)} tasks)\n")


def main() -> None:
    _IRA_DIR.mkdir(parents=True, exist_ok=True)
    (_IRA_DIR / "exports").mkdir(exist_ok=True)

    print("Creating Stage 1 IRA tasks …")
    create_stage1_tasks()

    print("Creating Stage 2 IRA tasks …")
    create_stage2_tasks()

    print("Done.")
    print()
    print("Next steps:")
    print("  1. Import data/ira/stage1_ira_tasks.json into your Stage 1 Label Studio project")
    print("  2. Import data/ira/stage2_ira_tasks.json into your Stage 2 Label Studio project")
    print("  3. Share the annotation_guide.md with your second rater")
    print("  4. After both raters annotate, export both sets to data/ira/exports/")
    print("     Name them: stage1_rater_a.json, stage1_rater_b.json,")
    print("                stage2_rater_a.json, stage2_rater_b.json")
    print("  5. Run: python scripts/ira/compute_ira.py")


if __name__ == "__main__":
    main()
