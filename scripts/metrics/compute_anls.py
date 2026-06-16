"""
Compute ANLS* scores for canonical Stage-2 approaches from raw predictions.

Uses the anls_star package (Peer et al., 2024) which implements the original
ANLS metric (Biten et al., ICCV 2019) with partial credit and Hungarian
matching for multi-value fields.

Two modes are computed:
  anls_all   : standard ANLS* over all emails (including emails where the field
               is absent; correct non-prediction scores 1.0 — inflates scores
               for rare fields like ATTACHMENT)
  anls_present: ANLS* restricted to emails where the ground-truth field is
               present (non-empty). Measures extraction quality only where
               extraction is actually required. Reported in the thesis.

Run after evaluate.py has produced stage2_raw_predictions.json with all
approaches present.

Usage:
    python scripts/metrics/compute_anls.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from anls_star import anls_score

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

DATA_DIR   = _ROOT / "data"
RESULTS    = DATA_DIR / "results"
STAGE2_ANN = DATA_DIR / "annotations" / "stage2_test.json"
RAW_PREDS  = RESULTS / "stage2_raw_predictions.json"

EVAL_FIELDS = ["FROM", "TO", "CC", "DATE", "SUBJECT", "ATTACHMENT"]
LIST_FIELDS = {"CC", "ATTACHMENT"}  # multi-value → use list ANLS matching

STAGE2_ORDER = [
    "Regex",
    "RobBERT (token, unweighted)",
    "RobBERT (token, class-weighted)",
    "GPT-5.5 (text)",
    "GPT-5.5 (vision+text)",
]

STAGE2_LABEL_ALIASES = {
    "BERT": "RobBERT (token, unweighted)",
    "BERT (weighted)": "RobBERT (token, class-weighted)",
    "GPT-5.5 v3 (text)": "GPT-5.5 (text)",
    "GPT-5.5 v3 (vision)": "GPT-5.5 (vision+text)",
}


def _normalize_approach(label: str) -> str:
    return STAGE2_LABEL_ALIASES.get(label, label)


def _gt_values(record: dict) -> dict[str, str | list[str]]:
    """Ground truth as {field: value} or {field: [values]} for list fields."""
    text = record["text"]
    by_field: dict[str, list[str]] = defaultdict(list)
    for f in record["fields"]:
        if f["label"] in EVAL_FIELDS:
            by_field[f["label"]].append(text[f["start_char"]:f["end_char"]].strip())
    result = {}
    for field in EVAL_FIELDS:
        vals = by_field.get(field, [])
        if field in LIST_FIELDS:
            result[field] = vals          # list (possibly empty)
        else:
            result[field] = vals[0] if vals else None  # scalar or None
    return result


def _pred_values(predictions: list[dict]) -> dict[str, str | list[str]]:
    """Predictions as {field: value} or {field: [values]}."""
    by_field: dict[str, list[str]] = defaultdict(list)
    for p in predictions:
        if p["label"] in EVAL_FIELDS:
            by_field[p["label"]].append(p["value"])
    result = {}
    for field in EVAL_FIELDS:
        vals = by_field.get(field, [])
        if field in LIST_FIELDS:
            result[field] = vals
        else:
            result[field] = vals[0] if vals else None
    return result


def _field_is_present(gt_val) -> bool:
    """True if the ground-truth field is non-empty (field exists in this email)."""
    if gt_val is None:
        return False
    if isinstance(gt_val, list):
        return len(gt_val) > 0
    return True


def compute_anls_scores(
    raw_preds: list[dict],
    gt_map: dict[tuple[str, str], dict],
    present_only: bool = False,
) -> dict[str, dict[str, float]]:
    """
    Returns {approach: {field: anls_score, "macro_anls": float}}.

    Parameters
    ----------
    present_only : if True, only evaluate on emails where the ground-truth
                   field is non-empty. Eliminates true-negative inflation for
                   rare fields (e.g. ATTACHMENT absent in 89% of emails).
                   If False, computes standard ANLS* over all emails.
    """
    by_approach: dict[str, dict[tuple, list]] = defaultdict(dict)
    for entry in raw_preds:
        key = (entry["dossier_id"], entry["email_id"])
        by_approach[_normalize_approach(entry["approach"])][key] = entry["predictions"]

    results = {}
    for approach, email_preds in by_approach.items():
        field_scores: dict[str, list[float]] = defaultdict(list)

        for key, preds in email_preds.items():
            if key not in gt_map:
                continue
            gt   = _gt_values(gt_map[key])
            pred = _pred_values(preds)

            for field in EVAL_FIELDS:
                gt_val   = gt[field]
                pred_val = pred.get(field, [] if field in LIST_FIELDS else None)

                if present_only and not _field_is_present(gt_val):
                    continue  # skip emails where field is absent

                score = anls_score(gt_val, pred_val)
                field_scores[field].append(score)

        approach_result = {}
        all_scores = []
        for field in EVAL_FIELDS:
            scores = field_scores[field]
            avg = round(sum(scores) / len(scores), 4) if scores else 0.0
            approach_result[field] = avg
            all_scores.extend(scores)
        approach_result["macro_anls"] = round(
            sum(approach_result[f] for f in EVAL_FIELDS) / len(EVAL_FIELDS), 4
        )
        approach_result["n_evaluated"] = len(all_scores)
        results[approach] = approach_result

    return results


def main() -> None:
    records   = json.load(open(STAGE2_ANN, encoding="utf-8"))
    gt_map    = {(r["dossier_id"], r["email_id"]): r for r in records}
    raw_preds = json.load(open(RAW_PREDS, encoding="utf-8"))

    approaches_present = sorted({_normalize_approach(e["approach"]) for e in raw_preds})
    print(f"Approaches in raw predictions: {approaches_present}\n")

    all_results     = compute_anls_scores(raw_preds, gt_map, present_only=False)
    present_results = compute_anls_scores(raw_preds, gt_map, present_only=True)

    def _print_table(results, title):
        print(f"=== {title} ===")
        header = f"{'Approach':<28} {'Macro':>7} " + "  ".join(f"{f[:6]:>8}" for f in EVAL_FIELDS)
        print(header)
        print("-" * len(header))
        for approach in STAGE2_ORDER:
            if approach not in results:
                continue
            r = results[approach]
            row = f"{approach:<28} {r['macro_anls']:>7.4f}  "
            row += "  ".join(f"{r[f]:>8.4f}" for f in EVAL_FIELDS)
            print(row)
        print()

    _print_table(all_results,     "ANLS* (all emails, including correct absences)")
    _print_table(present_results, "ANLS* (present-field emails only — reported in thesis)")

    # Write combined CSV: one row per approach with both anls_all and anls_present columns
    out_path = RESULTS / "stage2_anls.csv"
    rows = []
    for approach in STAGE2_ORDER:
        if approach not in all_results:
            continue
        ra = all_results[approach]
        rp = present_results.get(approach, {})
        row = {"approach": approach}
        row["anls_all_macro"] = ra["macro_anls"]
        for f in EVAL_FIELDS:
            row[f"anls_all_{f}"] = ra[f]
        row["anls_present_macro"] = rp.get("macro_anls", "")
        for f in EVAL_FIELDS:
            row[f"anls_present_{f}"] = rp.get(f, "")
        rows.append(row)

    fieldnames = (
        ["approach", "anls_all_macro"] +
        [f"anls_all_{f}" for f in EVAL_FIELDS] +
        ["anls_present_macro"] +
        [f"anls_present_{f}" for f in EVAL_FIELDS]
    )
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Written to {out_path.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
