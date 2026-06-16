"""
Compare RobBERT (line) Stage-1 predictions with and without the
merge_nearby_starts() proximity filter applied as post-processing.

Re-uses the unfiltered spans already saved in
data/results/stage1_raw_predictions.json (no model re-run needed) and
applies merge_nearby_starts() from src/bert_s1_predict.py to produce the
"filtered" variant for comparison.

This is a standalone comparison — it does NOT overwrite any existing
results files. Output is printed and written to
data/results/stage1_proximity_filter_comparison.csv.

Usage
-----
    python scripts/metrics/compare_proximity_filter.py
    python scripts/metrics/compare_proximity_filter.py --max-lines 8
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from eval_metrics import compute_span_metrics

# Inlined from src/bert_s1_predict.py to avoid importing torch (not needed here —
# this script only post-processes already-saved spans).
def merge_nearby_starts(
    spans: list[tuple[int, int]],
    text: str,
    max_lines: int = 8,
) -> list[tuple[int, int]]:
    """
    Post-processing filter: collapse consecutive predicted email starts that
    fall within max_lines of each other into just the first one.
    """
    if not spans:
        return spans

    filtered = [spans[0]]
    for start, end in spans[1:]:
        prev_start = filtered[-1][0]
        lines_between = text[prev_start:start].count("\n")
        if lines_between > max_lines:
            filtered.append((start, end))
        else:
            filtered[-1] = (filtered[-1][0], end)
    return filtered

DATA_DIR = _ROOT / "data"
STAGE1_TEST_ANN = DATA_DIR / "annotations" / "stage1_test.json"
RAW_PREDS = DATA_DIR / "results" / "stage1_raw_predictions.json"
OUT_CSV = DATA_DIR / "results" / "stage1_proximity_filter_comparison.csv"

UNFILTERED_LABEL = "RobBERT (line, unfiltered)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-lines", type=int, default=8)
    args = parser.parse_args()

    test_data = json.load(open(STAGE1_TEST_ANN, encoding="utf-8"))
    texts = {d["dossier_id"]: d["text"] for d in test_data}
    gt_spans = {
        d["dossier_id"]: [(e["start_char"], e["end_char"]) for e in d["emails"]]
        for d in test_data
    }

    raw = json.load(open(RAW_PREDS, encoding="utf-8"))

    rows = []
    unfiltered_f1s, filtered_f1s = [], []
    unfiltered_n, filtered_n = [], []

    print(f"{'Dossier':<12} {'n_true':>6} {'n_pred (raw)':>12} {'n_pred (filtered)':>18} "
          f"{'F1 (raw)':>9} {'F1 (filtered)':>14}")
    print("-" * 75)

    for did, text in texts.items():
        if did not in raw or UNFILTERED_LABEL not in raw[did]:
            continue
        spans_raw = [tuple(s) for s in raw[did][UNFILTERED_LABEL]]
        spans_filtered = merge_nearby_starts(spans_raw, text, max_lines=args.max_lines)

        m_raw = compute_span_metrics(spans_raw, gt_spans[did])
        m_filt = compute_span_metrics(spans_filtered, gt_spans[did])

        unfiltered_f1s.append(m_raw["f1"])
        filtered_f1s.append(m_filt["f1"])
        unfiltered_n.append(len(spans_raw))
        filtered_n.append(len(spans_filtered))

        print(f"{did:<12} {len(gt_spans[did]):>6} {len(spans_raw):>12} {len(spans_filtered):>18} "
              f"{m_raw['f1']:>9.4f} {m_filt['f1']:>14.4f}")

        rows.append({
            "dossier_id": did,
            "n_true": len(gt_spans[did]),
            "n_pred_raw": len(spans_raw),
            "n_pred_filtered": len(spans_filtered),
            "precision_raw": m_raw["precision"], "recall_raw": m_raw["recall"], "f1_raw": m_raw["f1"],
            "precision_filtered": m_filt["precision"], "recall_filtered": m_filt["recall"], "f1_filtered": m_filt["f1"],
        })

    n = len(rows)
    macro_raw = {
        "precision": round(sum(r["precision_raw"] for r in rows) / n, 4),
        "recall": round(sum(r["recall_raw"] for r in rows) / n, 4),
        "f1": round(sum(r["f1_raw"] for r in rows) / n, 4),
    }
    macro_filt = {
        "precision": round(sum(r["precision_filtered"] for r in rows) / n, 4),
        "recall": round(sum(r["recall_filtered"] for r in rows) / n, 4),
        "f1": round(sum(r["f1_filtered"] for r in rows) / n, 4),
    }

    print("-" * 75)
    print(f"{'MACRO':<12} {sum(r['n_true'] for r in rows):>6} "
          f"{sum(unfiltered_n):>12} {sum(filtered_n):>18} "
          f"{macro_raw['f1']:>9.4f} {macro_filt['f1']:>14.4f}")
    print()
    print(f"Raw      : P={macro_raw['precision']:.4f}  R={macro_raw['recall']:.4f}  F1={macro_raw['f1']:.4f}")
    print(f"Filtered : P={macro_filt['precision']:.4f}  R={macro_filt['recall']:.4f}  F1={macro_filt['f1']:.4f}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWritten to {OUT_CSV.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
