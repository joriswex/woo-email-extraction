"""
Convert Label Studio annotation exports to the stage1 / stage2 JSON formats
used by the training and evaluation pipelines.

Usage
-----
    # Stage 1 (email boundary detection)
    python scripts/convert_annotations.py \
        --stage1 data/annotations/project-3-at-*.json

    # Stage 2 (field extraction) — 80/20 train/test split by default
    python scripts/convert_annotations.py \
        --stage2 data/annotations/project-4-at-*.json

    # Custom test ratio (e.g. 15%)
    python scripts/convert_annotations.py \
        --stage2 data/annotations/project-4-at-*.json --test-ratio 0.15

    # Both stages at once
    python scripts/convert_annotations.py \
        --stage1 data/annotations/project-3-at-*.json \
        --stage2 data/annotations/project-4-at-*.json

Output
------
    data/annotations/stage1.json
    data/annotations/stage2_train.json
    data/annotations/stage2_test.json

The split is always freshly computed from a fixed seed (42) so it is
deterministic and reproducible. Delete the existing files before running
if you want to start from scratch with new data.

NOTE: never run evaluate.py on stage2_test.json until you are writing up
final results — treat it as strictly held-out.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from bert_s2_data import classify_email_type

ANN_DIR = _ROOT / "data" / "annotations"

STAGE2_FIELD_LABELS = {"FROM", "TO", "CC", "DATE", "SUBJECT", "ATTACHMENT", "REDACTED"}

DEFAULT_TEST_RATIO = 0.50
RANDOM_SEED = 42  # controls train/test split — change only with good reason


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_export(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _best_annotation(task: dict) -> list[dict]:
    """Return the result list of the first non-cancelled annotation."""
    for ann in task.get("annotations", []):
        if not ann.get("was_cancelled", False):
            return ann.get("result", [])
    return []


def _write(path: Path, data: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  → {path.relative_to(_ROOT)}  ({len(data)} records)")


# ---------------------------------------------------------------------------
# Stage 1 conversion
# ---------------------------------------------------------------------------


def convert_stage1(export_path: Path) -> None:
    print(f"Converting stage-1 export: {export_path.name}")
    tasks = _load_export(export_path)

    records = []
    for task in tasks:
        dossier_id = task["data"]["dossier_id"]
        text = task["data"]["text"]
        result = _best_annotation(task)

        email_spans = sorted(
            (r for r in result if r.get("value", {}).get("labels") == ["EMAIL"]),
            key=lambda r: r["value"]["start"],
        )
        emails = [
            {
                "email_id": f"email_{i}",
                "start_char": r["value"]["start"],
                "end_char": r["value"]["end"],
            }
            for i, r in enumerate(email_spans)
        ]

        records.append({"dossier_id": dossier_id, "text": text, "emails": emails})
        print(f"  {dossier_id}: {len(emails)} emails annotated")

    # Write full file (used by evaluate.py for GPT-4o cross-reference index)
    _write(ANN_DIR / "stage1.json", records)

    # Deterministic train / test split by dossier (parallel to stage 2)
    rng = random.Random(RANDOM_SEED)
    shuffled = records[:]
    rng.shuffle(shuffled)
    n_test  = max(1, round(len(shuffled) * DEFAULT_TEST_RATIO))
    n_train = len(shuffled) - n_test
    train, test = shuffled[:n_train], shuffled[n_train:]
    train_ids = [r["dossier_id"] for r in train]
    test_ids  = [r["dossier_id"] for r in test]
    print(f"  {len(records)} dossiers  →  {n_train} train  /  {n_test} test  "
          f"({DEFAULT_TEST_RATIO:.0%} test, seed={RANDOM_SEED})")
    print(f"  train: {train_ids}")
    print(f"  test : {test_ids}")
    _write(ANN_DIR / "stage1_train.json", train)
    _write(ANN_DIR / "stage1_test.json",  test)


# ---------------------------------------------------------------------------
# Stage 2 conversion
# ---------------------------------------------------------------------------


def convert_stage2(export_path: Path, test_ratio: float = DEFAULT_TEST_RATIO) -> None:
    print(f"Converting stage-2 export: {export_path.name}")
    tasks = _load_export(export_path)

    records = []
    for task in tasks:
        dossier_id = task["data"]["dossier_id"]
        email_id   = task["data"]["email_id"]
        text       = task["data"]["text"]
        result     = _best_annotation(task)

        fields = [
            {
                "label":      r["value"]["labels"][0],
                "start_char": r["value"]["start"],
                "end_char":   r["value"]["end"],
            }
            for r in sorted(result, key=lambda x: x["value"]["start"])
            if r["value"].get("labels", [None])[0] in STAGE2_FIELD_LABELS
        ]

        records.append({
            "dossier_id": dossier_id,
            "email_id":   email_id,
            "text":       text,
            "email_type": classify_email_type(text),
            "fields":     fields,
        })

    # Stratified train / test split by dossier_id — each dossier contributes
    # test_ratio of its emails to test, so all dossiers appear in both splits.
    rng = random.Random(RANDOM_SEED)
    from collections import defaultdict
    by_dossier: dict[str, list] = defaultdict(list)
    for r in records:
        by_dossier[r["dossier_id"]].append(r)

    train, test = [], []
    for did, recs in sorted(by_dossier.items()):
        shuffled = recs[:]
        rng.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * test_ratio))
        test.extend(shuffled[:n_test])
        train.extend(shuffled[n_test:])

    n_train, n_test = len(train), len(test)
    print(f"  {len(records)} total  →  {n_train} train  /  {n_test} test  "
          f"({test_ratio:.0%} test, stratified by dossier, seed={RANDOM_SEED})")
    _write(ANN_DIR / "stage2_train.json", train)
    _write(ANN_DIR / "stage2_test.json",  test)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Label Studio exports to stage1/stage2 annotation format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stage1", metavar="PATH", help="Label Studio export for stage 1")
    parser.add_argument("--stage2", metavar="PATH", help="Label Studio export for stage 2")
    parser.add_argument(
        "--test-ratio", type=float, default=DEFAULT_TEST_RATIO, metavar="FLOAT",
        help=f"Fraction of stage-2 data to reserve as test set (default: {DEFAULT_TEST_RATIO})",
    )
    args = parser.parse_args()

    if not args.stage1 and not args.stage2:
        parser.error("Provide at least --stage1 or --stage2.")

    if args.stage1:
        convert_stage1(Path(args.stage1))
    if args.stage2:
        convert_stage2(Path(args.stage2), test_ratio=args.test_ratio)

    print("\nDone.")


if __name__ == "__main__":
    main()
