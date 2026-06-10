"""
Convert Label Studio annotation exports to the stage1 / stage2 JSON formats
used by the training and evaluation pipelines.

Usage
-----
    # Stage 1 (email boundary detection) — one or more exports
    python scripts/convert_annotations.py \
        --stage1 data/annotations/project-3-at-*.json

    # Stage 2 (field extraction) — one or more exports
    python scripts/convert_annotations.py \
        --stage2 data/annotations/project-6-at-*.json

    # Rebuild splits from existing master files without adding new data
    python scripts/convert_annotations.py --rebuild-stage1
    python scripts/convert_annotations.py --rebuild-stage2

    # Both stages at once
    python scripts/convert_annotations.py \
        --stage1 data/annotations/project-3-at-*.json \
        --stage2 data/annotations/project-6-at-*.json

Output
------
    data/annotations/stage1.json          — complete stage-1 dataset (all dossiers)
    data/annotations/stage1_train.json    — train split (greedy email-count balanced)
    data/annotations/stage1_test.json     — test split  (greedy email-count balanced)
    data/annotations/stage2.json          — complete stage-2 dataset (all emails)
    data/annotations/stage2_train.json    — train split (stratified by email within dossier)
    data/annotations/stage2_test.json     — test split  (stratified by email within dossier)

Merging behaviour
-----------------
Both converters are additive: if a master file (stage1.json / stage2.json) already
exists, its records are kept and records from the new export are merged in.
For stage 1, an incoming record with the same dossier_id replaces the existing one
(re-annotation takes precedence).  For stage 2, the key is (dossier_id, email_id).

On first run, if stage2.json does not exist yet, the converter bootstraps from any
existing stage2_train.json / stage2_test.json so that previously converted data is
never lost.  The same fallback applies to stage1_train.json / stage1_test.json for
stage 1.

Stage-1 split strategy
-----------------------
Dossiers vary widely in size (13–333 emails in the current corpus).  A plain random
50/50 split of 20 dossiers could easily be skewed by hundreds of emails.  Instead we
use a greedy email-count balancing algorithm:

    1. Sort dossiers by email count, largest first.
    2. Assign each dossier to whichever split currently has fewer total emails
       (ties go to train).

This is fully deterministic (no random seed needed) and produces near-perfectly
balanced splits regardless of the size distribution — e.g. 1 007 vs 1 006 emails
across 10 / 10 dossiers for the current 20-dossier corpus.

Stage-2 split strategy
-----------------------
Each dossier contributes ~50 % of its emails to the test split (stratified within
dossier, seeded for reproducibility).  This ensures every dossier appears in both
train and test and that the overall split ratio is consistent regardless of how
much each individual dossier contributed.

NOTE: never run evaluate.py on stage*_test.json until you are writing up
final results — treat them as strictly held-out.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from bert_s2_data import classify_email_type

ANN_DIR = _ROOT / "data" / "annotations"

STAGE2_FIELD_LABELS = {"FROM", "TO", "CC", "DATE", "SUBJECT", "ATTACHMENT"}

DEFAULT_TEST_RATIO = 0.50
RANDOM_SEED = 42  # controls stage-2 stratified split — change only with good reason


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_export(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_existing(path: Path) -> list[dict]:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return []


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
# Stage-1 split: greedy email-count balancing
# ---------------------------------------------------------------------------


def _greedy_split(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Split dossier records into two halves balanced by total email count.

    Algorithm
    ---------
    Sort dossiers by email count descending, then assign each to whichever
    side currently has fewer emails (ties go to train).  This greedy approach
    is fully deterministic and produces near-perfectly balanced splits for any
    size distribution — no random seed required.

    This is preferable to a random shuffle for stage 1 because the corpus
    contains dossiers ranging from 13 to 333 emails: the two largest alone
    account for ~30 % of all emails, so a random 10/10 split could easily be
    skewed by 200–300 emails.  The greedy method eliminates that risk.
    """
    sorted_records = sorted(records, key=lambda r: len(r["emails"]), reverse=True)

    train: list[dict] = []
    test: list[dict] = []
    train_emails = 0
    test_emails = 0

    for record in sorted_records:
        n = len(record["emails"])
        if train_emails <= test_emails:
            train.append(record)
            train_emails += n
        else:
            test.append(record)
            test_emails += n

    return train, test


# ---------------------------------------------------------------------------
# Stage 1 conversion
# ---------------------------------------------------------------------------


def convert_stage1(export_paths: list[Path]) -> None:
    # --- Convert new exports (if any) ---
    new_records: dict[str, dict] = {}
    for export_path in export_paths:
        print(f"Converting stage-1 export: {export_path.name}")
        for task in _load_export(export_path):
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
            new_records[dossier_id] = {"dossier_id": dossier_id, "text": text, "emails": emails}
            print(f"  {dossier_id}: {len(emails)} emails annotated")

    # --- Load existing master file ---
    existing: dict[str, dict] = {
        r["dossier_id"]: r for r in _load_existing(ANN_DIR / "stage1.json")
    }

    # Recover any dossiers that exist only in the train/test split files but
    # are missing from stage1.json (e.g. after an accidental overwrite).
    for path in (ANN_DIR / "stage1_train.json", ANN_DIR / "stage1_test.json"):
        for r in _load_existing(path):
            if r["dossier_id"] not in existing:
                existing[r["dossier_id"]] = r
                print(f"  Recovered {r['dossier_id']} from {path.name}")

    # New export wins on conflict — re-annotation takes precedence
    merged = {**existing, **new_records}
    all_records = list(merged.values())
    n_existing = len(existing) - len(set(existing) & set(new_records))
    print(f"  Merged: {n_existing} existing + {len(new_records)} from export"
          f" = {len(all_records)} total dossiers")

    _write(ANN_DIR / "stage1.json", all_records)

    # --- Greedy email-count balanced split ---
    train, test = _greedy_split(all_records)
    train_email_count = sum(len(r["emails"]) for r in train)
    test_email_count  = sum(len(r["emails"]) for r in test)
    print(f"  {len(all_records)} dossiers  →  "
          f"{len(train)} train ({train_email_count} emails) / "
          f"{len(test)} test ({test_email_count} emails)  "
          f"[greedy email-count balanced]")
    _write(ANN_DIR / "stage1_train.json", train)
    _write(ANN_DIR / "stage1_test.json",  test)


# ---------------------------------------------------------------------------
# Stage 2 conversion
# ---------------------------------------------------------------------------


def convert_stage2(export_paths: list[Path], test_ratio: float = DEFAULT_TEST_RATIO) -> None:
    # --- Convert new exports (if any) ---
    new_records: dict[tuple[str, str], dict] = {}
    for export_path in export_paths:
        print(f"Converting stage-2 export: {export_path.name}")
        for task in _load_export(export_path):
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

            new_records[(dossier_id, email_id)] = {
                "dossier_id": dossier_id,
                "email_id":   email_id,
                "text":       text,
                "email_type": classify_email_type(text),
                "fields":     fields,
            }

    # --- Load existing master file ---
    existing_raw = _load_existing(ANN_DIR / "stage2.json")

    # Bootstrap from split files if stage2.json does not exist yet.
    # This handles the initial migration from runs before the master file was introduced.
    if not existing_raw:
        train_recs = _load_existing(ANN_DIR / "stage2_train.json")
        test_recs  = _load_existing(ANN_DIR / "stage2_test.json")
        if train_recs or test_recs:
            existing_raw = train_recs + test_recs
            print(f"  Bootstrapping stage2.json from existing train/test files "
                  f"({len(existing_raw)} records)")

    existing: dict[tuple[str, str], dict] = {
        (r["dossier_id"], r["email_id"]): r for r in existing_raw
    }

    # New export wins on conflict — re-annotation takes precedence
    merged = {**existing, **new_records}
    all_records = list(merged.values())
    n_existing = len(existing) - len(set(existing) & set(new_records))
    print(f"  Merged: {n_existing} existing + {len(new_records)} from export"
          f" = {len(all_records)} total records")

    _write(ANN_DIR / "stage2.json", all_records)

    # --- Stratified split: each dossier contributes test_ratio of its emails to test ---
    # Splitting within each dossier (rather than assigning whole dossiers) ensures all
    # dossiers appear in both train and test and that the split ratio is consistent
    # regardless of individual dossier sizes.
    rng = random.Random(RANDOM_SEED)
    by_dossier: dict[str, list] = defaultdict(list)
    for r in all_records:
        by_dossier[r["dossier_id"]].append(r)

    train, test = [], []
    for did, recs in sorted(by_dossier.items()):
        shuffled = recs[:]
        rng.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * test_ratio))
        test.extend(shuffled[:n_test])
        train.extend(shuffled[n_test:])

    print(f"  {len(all_records)} total  →  {len(train)} train / {len(test)} test  "
          f"({test_ratio:.0%} test, stratified by email within dossier, seed={RANDOM_SEED})")
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
    parser.add_argument(
        "--stage1", metavar="PATH", nargs="+",
        help="Label Studio export file(s) for stage 1",
    )
    parser.add_argument(
        "--stage2", metavar="PATH", nargs="+",
        help="Label Studio export file(s) for stage 2",
    )
    parser.add_argument(
        "--rebuild-stage1", action="store_true",
        help="Re-merge and re-split stage 1 from existing files without adding new data",
    )
    parser.add_argument(
        "--rebuild-stage2", action="store_true",
        help="Re-merge and re-split stage 2 from existing files without adding new data",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=DEFAULT_TEST_RATIO, metavar="FLOAT",
        help=f"Fraction of stage-2 data to reserve as test set (default: {DEFAULT_TEST_RATIO})",
    )
    args = parser.parse_args()

    if not any([args.stage1, args.stage2, args.rebuild_stage1, args.rebuild_stage2]):
        parser.error("Provide at least one of --stage1, --stage2, --rebuild-stage1, --rebuild-stage2.")

    if args.stage1 or args.rebuild_stage1:
        convert_stage1([Path(p) for p in (args.stage1 or [])])
    if args.stage2 or args.rebuild_stage2:
        convert_stage2([Path(p) for p in (args.stage2 or [])], test_ratio=args.test_ratio)

    print("\nDone.")


if __name__ == "__main__":
    main()
