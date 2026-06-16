"""
Success/failure analysis for Stage 1 (email boundary detection) and
Stage 2 (field extraction).

For each unit of evaluation (a ground-truth email span in Stage 1, or a
(email, field) pair with n_gt > 0 in Stage 2), determines for each
approach whether it got that unit fully correct, then categorizes:

  - all_correct : every approach got it right
  - all_fail    : every approach got it wrong
  - mixed       : some approaches right, some wrong

Prints aggregate counts/rates and a handful of example "mixed" and
"all_fail" cases (with predicted vs. ground-truth values) to help spot
patterns in what breaks which approaches.

Usage
-----
    python scripts/analyze_success_failure.py stage1
    python scripts/analyze_success_failure.py stage2
    python scripts/analyze_success_failure.py both
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from eval_metrics import exact_match, _greedy_tp  # noqa: E402

DATA_DIR = _ROOT / "data"
RESULTS = DATA_DIR / "results"

STAGE1_TEST = DATA_DIR / "annotations" / "stage1_test.json"
STAGE1_PREDS = RESULTS / "stage1_raw_predictions.json"

STAGE2_TEST = DATA_DIR / "annotations" / "stage2_test.json"
STAGE2_PREDS = RESULTS / "stage2_raw_predictions.json"

EVAL_FIELDS = ["FROM", "TO", "CC", "DATE", "SUBJECT", "ATTACHMENT"]

STAGE1_APPROACHES = ["Regex", "RobBERT (line + proximity filter)", "GPT-5.5 (text)", "GPT-5.5 (vision+text)"]
STAGE2_APPROACHES = ["Regex", "BERT", "BERT (weighted)", "GPT-5.5 (text)", "GPT-5.5 (vision+text)"]


def iou(a, b) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def stage1():
    test = json.load(open(STAGE1_TEST, encoding="utf-8"))
    preds = json.load(open(STAGE1_PREDS, encoding="utf-8"))

    cat_counts = Counter()
    examples = defaultdict(list)

    for record in test:
        did = record["dossier_id"]
        text = record["text"]
        for email in record["emails"]:
            gt_span = (email["start_char"], email["end_char"])
            results = {}
            for approach in STAGE1_APPROACHES:
                spans = preds[did].get(approach, [])
                hit = any(iou(ps, gt_span) >= 0.5 for ps in spans)
                results[approach] = hit

            n_correct = sum(results.values())
            if n_correct == len(STAGE1_APPROACHES):
                cat = "all_correct"
            elif n_correct == 0:
                cat = "all_fail"
            else:
                cat = "mixed"
            cat_counts[cat] += 1

            snippet = text[gt_span[0]:gt_span[1]][:200].replace("\n", " \\n ")
            examples[cat].append({
                "dossier": did,
                "email_id": email["email_id"],
                "span": gt_span,
                "len": gt_span[1] - gt_span[0],
                "results": results,
                "snippet": snippet,
            })

    total = sum(cat_counts.values())
    print("=== Stage 1: per ground-truth email, did each approach IoU>=0.5 hit it? ===")
    for cat in ["all_correct", "mixed", "all_fail"]:
        n = cat_counts[cat]
        print(f"{cat:<12} {n:5d}  ({n/total:5.1%})")
    print()

    # which approaches fail most often within "mixed" cases
    fail_in_mixed = Counter()
    for ex in examples["mixed"]:
        for approach, ok in ex["results"].items():
            if not ok:
                fail_in_mixed[approach] += 1
    print("In 'mixed' cases, # times each approach was the one that FAILED:")
    for a, c in fail_in_mixed.most_common():
        print(f"  {a:<35} {c:4d} / {cat_counts['mixed']}")
    print()

    # length stats
    for cat in ["all_correct", "mixed", "all_fail"]:
        lens = [ex["len"] for ex in examples[cat]]
        if lens:
            print(f"{cat:<12} mean span length = {sum(lens)/len(lens):7.1f} chars (n={len(lens)})")
    print()

    print("Sample 'all_fail' cases (no approach found this email boundary):")
    for ex in examples["all_fail"][:5]:
        print(f"  dossier={ex['dossier']} {ex['email_id']} len={ex['len']}: {ex['snippet']!r}")
    print()

    print("Sample 'mixed' cases:")
    for ex in examples["mixed"][:8]:
        failed = [a for a, ok in ex["results"].items() if not ok]
        print(f"  dossier={ex['dossier']} {ex['email_id']} len={ex['len']} failed=[{', '.join(failed)}]: {ex['snippet']!r}")
    print()


def stage2():
    test = json.load(open(STAGE2_TEST, encoding="utf-8"))
    raw = json.load(open(STAGE2_PREDS, encoding="utf-8"))

    preds_by_key = defaultdict(list)
    for r in raw:
        preds_by_key[(r["dossier_id"], r["email_id"], r["approach"])] = r["predictions"]

    cat_counts = Counter()
    field_cat_counts = defaultdict(Counter)
    examples = defaultdict(list)
    fail_counter = defaultdict(Counter)  # field -> approach -> fail count in mixed

    for record in test:
        did = record["dossier_id"]
        eid = record["email_id"]
        text = record["text"]
        gt_by_field = defaultdict(list)
        for f in record["fields"]:
            if f["label"] in EVAL_FIELDS:
                gt_by_field[f["label"]].append(text[f["start_char"]:f["end_char"]].strip())

        for field, gts in gt_by_field.items():
            results = {}
            pred_values = {}
            for approach in STAGE2_APPROACHES:
                preds = preds_by_key.get((did, eid, approach), [])
                pred_vals = [p["value"] for p in preds if p["label"] == field]
                tp = _greedy_tp(pred_vals, gts, exact_match)
                results[approach] = (tp == len(gts))
                pred_values[approach] = pred_vals

            n_correct = sum(results.values())
            if n_correct == len(STAGE2_APPROACHES):
                cat = "all_correct"
            elif n_correct == 0:
                cat = "all_fail"
            else:
                cat = "mixed"
            cat_counts[cat] += 1
            field_cat_counts[field][cat] += 1

            if cat == "mixed":
                for approach, ok in results.items():
                    if not ok:
                        fail_counter[field][approach] += 1

            examples[(field, cat)].append({
                "dossier": did, "email_id": eid, "email_type": record.get("email_type"),
                "gt": gts, "preds": pred_values, "results": results,
            })

    total = sum(cat_counts.values())
    print("=== Stage 2: per (email, field), did each approach get exact match (full recall)? ===")
    for cat in ["all_correct", "mixed", "all_fail"]:
        n = cat_counts[cat]
        print(f"{cat:<12} {n:5d}  ({n/total:5.1%})")
    print()

    print("Per field breakdown:")
    print(f"{'Field':<12} {'all_correct':>12} {'mixed':>8} {'all_fail':>9}  total")
    for field in EVAL_FIELDS:
        c = field_cat_counts[field]
        t = sum(c.values())
        print(f"{field:<12} {c['all_correct']/t:11.1%} {c['mixed']/t:7.1%} {c['all_fail']/t:8.1%}  (n={t})")
    print()

    print("In 'mixed' cases, # times each approach was the one that FAILED, by field:")
    for field in EVAL_FIELDS:
        if field not in fail_counter:
            continue
        n_mixed = field_cat_counts[field]["mixed"]
        print(f"  {field} (n_mixed={n_mixed}):")
        for a, c in fail_counter[field].most_common():
            print(f"    {a:<20} {c:4d} / {n_mixed}")
    print()

    print("Sample 'all_fail' cases per field (no approach got exact match):")
    for field in EVAL_FIELDS:
        exs = examples[(field, "all_fail")]
        if not exs:
            continue
        print(f"  -- {field} --")
        for ex in exs[:3]:
            print(f"    dossier={ex['dossier']} {ex['email_id']} ({ex['email_type']})")
            print(f"      GT:    {ex['gt']}")
            for a in STAGE2_APPROACHES:
                print(f"      {a:<20}: {ex['preds'][a]}")
    print()

    print("Sample 'mixed' cases (ATTACHMENT and CC):")
    for field in ["ATTACHMENT", "CC"]:
        exs = examples[(field, "mixed")]
        if not exs:
            continue
        print(f"  -- {field} --")
        for ex in exs[:3]:
            failed = [a for a, ok in ex["results"].items() if not ok]
            print(f"    dossier={ex['dossier']} {ex['email_id']} failed={failed}")
            print(f"      GT:    {ex['gt']}")
            for a in STAGE2_APPROACHES:
                print(f"      {a:<20}: {ex['preds'][a]}")
    print()


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("stage1", "both"):
        stage1()
    if which in ("stage2", "both"):
        stage2()


if __name__ == "__main__":
    main()
