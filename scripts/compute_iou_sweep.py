"""
Sweep the IoU threshold for Stage 1 (email boundary detection) and report
macro-averaged precision/recall/F1 per approach at each threshold.

Re-uses the spans already saved in data/results/stage1_raw_predictions.json,
so no models need to be re-run (no GPT cost).

Usage
-----
    python scripts/compute_iou_sweep.py
    python scripts/compute_iou_sweep.py --thresholds 0.3 0.5 0.7 0.9 1.0
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from eval_metrics import compute_span_metrics

DATA_DIR = _ROOT / "data"
STAGE1_TEST_ANN = DATA_DIR / "annotations" / "stage1_test.json"   # 10-dossier official test set
RAW_PREDS = DATA_DIR / "results" / "stage1_raw_predictions.json"
OUT_CSV = DATA_DIR / "results" / "stage1_iou_sweep.csv"

DEFAULT_THRESHOLDS = [0.3, 0.5, 0.7, 0.9, 1.0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    args = parser.parse_args()

    test_dossiers = {d["dossier_id"] for d in json.load(open(STAGE1_TEST_ANN, encoding="utf-8"))}
    gt_data = json.load(open(STAGE1_TEST_ANN, encoding="utf-8"))
    gt_spans = {
        d["dossier_id"]: [(e["start_char"], e["end_char"]) for e in d["emails"]]
        for d in gt_data
    }

    raw = json.load(open(RAW_PREDS, encoding="utf-8"))

    # Collect approach -> dossier_id -> predicted spans
    by_approach: dict[str, dict[str, list]] = {}
    for did, apps in raw.items():
        for approach, spans in apps.items():
            by_approach.setdefault(approach, {})[did] = [tuple(s) for s in spans]

    rows = []
    print(f"{'Approach':<35} {'n_dossiers':>10}  " + "  ".join(f"F1@{t:.1f}" for t in args.thresholds))
    print("-" * (35 + 12 + 8 * len(args.thresholds)))
    for approach, dossiers in by_approach.items():
        # restrict to the official 10-dossier test set for a fair, apples-to-apples
        # comparison across all approaches.
        eval_dossiers = {did: pred for did, pred in dossiers.items() if did in test_dossiers}
        f1s_per_threshold = []
        for t in args.thresholds:
            f1s = []
            for did, pred in eval_dossiers.items():
                if did not in gt_spans:
                    continue
                m = compute_span_metrics(pred, gt_spans[did], iou_threshold=t)
                f1s.append(m["f1"])
                rows.append({
                    "approach": approach, "dossier_id": did, "iou_threshold": t,
                    "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                })
            f1s_per_threshold.append(round(sum(f1s) / len(f1s), 4) if f1s else 0.0)
        n = len(eval_dossiers)
        print(f"{approach:<35} {n:>10}  " + "  ".join(f"{f:.4f}" for f in f1s_per_threshold))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["approach", "dossier_id", "iou_threshold", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWritten to {OUT_CSV.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
