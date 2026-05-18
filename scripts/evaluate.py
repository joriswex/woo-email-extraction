"""
Evaluation script: Regex vs. BERT vs. GPT-5.5 on Woo dossier extraction.

Stage 1 — email boundary detection (dossier level)
  Approaches : Regex, BERT (line-level with proximity filter)
  Metric     : span IoU >= 0.5  ->  precision / recall / F1
  Output     : data/results/stage1_per_dossier.csv
               data/results/stage1_summary.csv

Stage 2 — field extraction (email level)
  Approaches : Regex, BERT, GPT-5.5 (vision)
  Metric     : value-level exact match and ANLS  ->  P/R/F1
  Output     : data/results/stage2_per_field.csv
               data/results/stage2_summary.csv

Usage
-----
    # Regex only (default — no API key needed)
    python scripts/evaluate.py

    # Add GPT-5.5 vision (requires OPENAI_API_KEY)
    python scripts/evaluate.py --gpt

    # Full comparison with BERT checkpoints
    python scripts/evaluate.py --gpt --bert-s1 models/stage1 --bert-s2 models/stage2

    # Stage 2 only with ANLS computation
    python scripts/evaluate.py --stages 2 --anls

    # Compute ANLS after evaluation
    python scripts/evaluate.py --gpt --anls
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from eval_metrics import (
    accumulate_field_counts,
    compute_span_metrics,
    empty_counts,
    macro_f1,
    prf_from_counts,
)
from regex_baseline import (
    detect_email_boundaries as regex_boundaries,
    extract_fields as regex_fields,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR   = _ROOT / "data"
STAGE1_ANN = DATA_DIR / "annotations" / "stage1_test.json"
STAGE2_ANN = DATA_DIR / "annotations" / "stage2_test.json"
RAW_DIR    = DATA_DIR / "raw"

EVAL_FIELDS = ["FROM", "TO", "CC", "DATE", "SUBJECT", "ATTACHMENT"]
DASH = "—"  # shown in CSV when an approach was not run

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

_STAGE1_ORDER = ["Regex", "BERT", "GPT-5.5 (vision)"]
_STAGE2_ORDER = ["Regex", "BERT", "GPT-5.5 (vision)"]


def _merge_write_csv(
    path: Path,
    new_rows: list[dict],
    fieldnames: list[str],
    run_approaches: list[str],
    canonical_order: list[str],
) -> None:
    """Write CSV rows, preserving scores for approaches not run this time."""
    kept: list[dict] = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            kept = [r for r in csv.DictReader(f) if r.get("approach") not in run_approaches]

    all_rows = kept + new_rows
    order_map = {a: i for i, a in enumerate(canonical_order)}
    all_rows.sort(key=lambda r: order_map.get(r.get("approach", ""), 999))

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    archive_dir = path.parent / "archive" / _RUN_TIMESTAMP
    archive_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, archive_dir / path.name)
    print(f"  -> {path.relative_to(_ROOT)}")


def _gt_values(record: dict) -> list[dict]:
    """Ground truth field values for a stage-2 record (REDACTED excluded)."""
    text = record["text"]
    return [
        {"label": f["label"], "value": text[f["start_char"]:f["end_char"]].strip()}
        for f in record["fields"]
        if f["label"] in EVAL_FIELDS
    ]


def _spans_to_values(spans: list[dict], text: str) -> list[dict]:
    """Convert span dicts (start_char/end_char) to value dicts."""
    return [
        {"label": s["label"], "value": text[s["start_char"]:s["end_char"]].strip()}
        for s in spans
        if s["label"] in EVAL_FIELDS
    ]


_PAGE_MAP_CACHE = DATA_DIR / "page_maps_cache.json"


def _build_page_maps(dossier_ids: list[str]) -> dict[str, dict]:
    """Build or load cached page maps for GPT-5.5 vision."""
    cache: dict[str, dict] = {}
    if _PAGE_MAP_CACHE.exists():
        with open(_PAGE_MAP_CACHE) as f:
            raw = json.load(f)
        cache = {did: {int(k): tuple(v) for k, v in pm.items()} for did, pm in raw.items()}

    missing = [did for did in dossier_ids if did not in cache]
    if missing:
        from pdf_extract import extract_text
        print(f"  Building page maps for {len(missing)} dossier(s) (cached afterwards) ...")
        for did in missing:
            pdf = RAW_DIR / f"{did}.pdf"
            if not pdf.exists():
                print(f"  WARNING: {did}.pdf not found, GPT-5.5 vision will be skipped")
                cache[did] = {}
                continue
            _, page_map = extract_text(pdf)
            cache[did] = page_map

        serialisable = {did: {str(k): list(v) for k, v in pm.items()} for did, pm in cache.items()}
        with open(_PAGE_MAP_CACHE, "w") as f:
            json.dump(serialisable, f)
        print(f"  Page maps cached to {_PAGE_MAP_CACHE.relative_to(_ROOT)}")
    else:
        print("  Page maps loaded from cache.")

    return {did: cache[did] for did in dossier_ids}


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------


def run_stage1(
    records: list[dict],
    bert_s1=None,
    bert_s1_tok=None,
    run_gpt: bool = True,
    output_dir: Path = DATA_DIR / "results",
) -> None:
    print("\n=== Stage 1: email boundary detection ===")

    per_dossier: list[dict] = []
    approach_metrics: dict[str, list[dict]] = defaultdict(list)
    raw_preds_s1: dict = {}

    for record in records:
        did = record["dossier_id"]
        text = record["text"]
        gt_spans = [(e["start_char"], e["end_char"]) for e in record["emails"]]
        print(f"  {did}  ({len(gt_spans)} emails)")

        raw_preds_s1[did] = {}

        # Regex
        r_spans = regex_boundaries(text)
        r_m = compute_span_metrics(r_spans, gt_spans)
        per_dossier.append({"approach": "Regex", "dossier_id": did, **r_m})
        approach_metrics["Regex"].append(r_m)
        raw_preds_s1[did]["Regex"] = r_spans

        # BERT (line-level with proximity filter)
        if bert_s1:
            from bert_s1_predict import predict, merge_nearby_starts
            b_spans = predict(text, bert_s1, bert_s1_tok)
            b_spans = merge_nearby_starts(b_spans, text)
            b_m = compute_span_metrics(b_spans, gt_spans)
            per_dossier.append({"approach": "BERT", "dossier_id": did, **b_m})
            approach_metrics["BERT"].append(b_m)
            raw_preds_s1[did]["BERT"] = b_spans

        # GPT-5.5 (vision) for stage 1 uses text-based boundary detection
        if run_gpt:
            import gpt_baseline
            try:
                g_spans = gpt_baseline.detect_email_boundaries_text(text)
                g_m = compute_span_metrics(g_spans, gt_spans)
                per_dossier.append({"approach": "GPT-5.5 (vision)", "dossier_id": did, **g_m})
                approach_metrics["GPT-5.5 (vision)"].append(g_m)
                raw_preds_s1[did]["GPT-5.5 (vision)"] = g_spans
            except Exception as exc:
                print(f"    GPT-5.5 error on {did}: {exc}")
                per_dossier.append({
                    "approach": "GPT-5.5 (vision)", "dossier_id": did,
                    "precision": DASH, "recall": DASH, "f1": DASH,
                    "n_pred": DASH, "n_true": len(gt_spans),
                })

    # Save raw predictions
    raw_s1_path = output_dir / "stage1_raw_predictions.json"
    raw_s1_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_s1_path, "w") as f:
        json.dump({did: {app: [list(s) for s in spans]
                         for app, spans in apps.items()}
                   for did, apps in raw_preds_s1.items()}, f)
    print(f"  -> {raw_s1_path.relative_to(_ROOT)}")

    # Summary: macro-average across dossiers
    summary: list[dict] = []
    for approach, metrics in approach_metrics.items():
        agg = {k: round(sum(m[k] for m in metrics) / len(metrics), 4)
               for k in ("precision", "recall", "f1")}
        summary.append({"approach": approach, **agg})

    run_approaches = list(approach_metrics.keys())

    _merge_write_csv(
        output_dir / "stage1_per_dossier.csv",
        per_dossier,
        ["approach", "dossier_id", "precision", "recall", "f1", "n_pred", "n_true"],
        run_approaches=run_approaches,
        canonical_order=_STAGE1_ORDER,
    )
    _merge_write_csv(
        output_dir / "stage1_summary.csv",
        summary,
        ["approach", "precision", "recall", "f1"],
        run_approaches=run_approaches,
        canonical_order=_STAGE1_ORDER,
    )


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------


def run_stage2(
    records: list[dict],
    stage1_index: dict,          # {dossier_id: {email_id: (start, end)}}
    page_maps: dict[str, dict],  # {dossier_id: page_map}
    bert_s2=None,
    bert_s2_tok=None,
    run_gpt: bool = True,
    output_dir: Path = DATA_DIR / "results",
) -> None:
    print(f"\n=== Stage 2: field extraction ({len(records)} emails) ===")

    approaches = ["Regex"]
    if bert_s2:
        approaches.append("BERT")
    if run_gpt:
        approaches.append("GPT-5.5 (vision)")

    counters: dict[str, dict] = {a: empty_counts() for a in approaches}
    raw_preds_s2: list[dict] = []

    for i, record in enumerate(records):
        did  = record["dossier_id"]
        eid  = record["email_id"]
        text = record["text"]
        gt   = _gt_values(record)

        # Regex
        r_preds = [{"label": p["label"], "value": p["value"]}
                   for p in regex_fields(text) if p["label"] in EVAL_FIELDS]
        accumulate_field_counts(counters["Regex"], r_preds, gt, EVAL_FIELDS)
        raw_preds_s2.append({"dossier_id": did, "email_id": eid,
                              "approach": "Regex", "predictions": r_preds})

        # BERT
        if bert_s2:
            from bert_s2_predict import predict_fields
            b_spans = predict_fields(text, bert_s2, bert_s2_tok)
            b_preds = _spans_to_values(b_spans, text)
            accumulate_field_counts(counters["BERT"], b_preds, gt, EVAL_FIELDS)
            raw_preds_s2.append({"dossier_id": did, "email_id": eid,
                                  "approach": "BERT", "predictions": b_preds})

        # GPT-5.5 (vision) — always prompt version 3
        if run_gpt:
            pos = stage1_index.get(did, {}).get(eid)
            pm  = page_maps.get(did, {})
            if pos and pm:
                import gpt_baseline
                try:
                    g_preds = gpt_baseline.extract_fields_vision(
                        RAW_DIR / f"{did}.pdf", pm, pos[0], pos[1],
                        email_text=text, prompt_version=3,
                    )
                    accumulate_field_counts(counters["GPT-5.5 (vision)"], g_preds, gt, EVAL_FIELDS)
                    raw_preds_s2.append({"dossier_id": did, "email_id": eid,
                                         "approach": "GPT-5.5 (vision)", "predictions": g_preds})
                except Exception as exc:
                    print(f"    GPT-5.5 error on {did}/{eid}: {exc}")
                time.sleep(0.3)
            else:
                print(f"    WARNING: no position found for {did}/{eid}, skipping GPT-5.5")

        if (i + 1) % 10 == 0 or (i + 1) == len(records):
            print(f"  {i + 1}/{len(records)} emails processed")

    run_approaches = list(counters.keys())

    # Save raw predictions (merge with existing to preserve other approaches)
    raw_s2_path = output_dir / "stage2_raw_predictions.json"
    existing_raw: list[dict] = []
    if raw_s2_path.exists():
        with open(raw_s2_path) as f:
            existing_raw = json.load(f)
    kept = [e for e in existing_raw if e["approach"] not in run_approaches]
    merged_raw = kept + raw_preds_s2
    with open(raw_s2_path, "w") as f:
        json.dump(merged_raw, f, ensure_ascii=False)
    print(f"  -> {raw_s2_path.relative_to(_ROOT)}")

    # Build per-field CSV rows
    per_field_rows: list[dict] = []
    summary_rows: list[dict] = []

    for approach in approaches:
        field_metrics = prf_from_counts(counters[approach])
        for field in EVAL_FIELDS:
            if field in field_metrics:
                per_field_rows.append({
                    "approach": approach,
                    "field": field,
                    **field_metrics[field],
                })
        mf = macro_f1(field_metrics)
        summary_rows.append({
            "approach": approach,
            "exact_macro_f1": mf["exact_f1"],
            "anls_macro_f1":  mf.get("anls_f1", ""),
            "n_emails": len(records),
        })

    _merge_write_csv(
        output_dir / "stage2_per_field.csv",
        per_field_rows,
        ["approach", "field", "exact_p", "exact_r", "exact_f1", "anls_p", "anls_r", "anls_f1"],
        run_approaches=run_approaches,
        canonical_order=_STAGE2_ORDER,
    )
    _merge_write_csv(
        output_dir / "stage2_summary.csv",
        summary_rows,
        ["approach", "exact_macro_f1", "anls_macro_f1", "n_emails"],
        run_approaches=run_approaches,
        canonical_order=_STAGE2_ORDER,
    )


# ---------------------------------------------------------------------------
# ANLS computation (folded in from compute_anls.py)
# ---------------------------------------------------------------------------


def _gt_anls_values(record: dict) -> dict[str, str | list[str]]:
    """Ground truth as {field: value} or {field: [values]} for list fields."""
    LIST_FIELDS = {"CC", "ATTACHMENT"}
    text = record["text"]
    by_field: dict[str, list[str]] = defaultdict(list)
    for f in record["fields"]:
        if f["label"] in EVAL_FIELDS:
            by_field[f["label"]].append(text[f["start_char"]:f["end_char"]].strip())
    result = {}
    for field in EVAL_FIELDS:
        vals = by_field.get(field, [])
        if field in LIST_FIELDS:
            result[field] = vals
        else:
            result[field] = vals[0] if vals else None
    return result


def _pred_anls_values(predictions: list[dict]) -> dict[str, str | list[str]]:
    """Predictions as {field: value} or {field: [values]}."""
    LIST_FIELDS = {"CC", "ATTACHMENT"}
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


def run_anls(
    output_dir: Path = DATA_DIR / "results",
) -> None:
    """
    Compute ANLS* scores for all approaches from stage2_raw_predictions.json.

    Uses the anls_star package (Peer et al., 2024) which implements the original
    ANLS metric (Biten et al., ICCV 2019) with partial credit and Hungarian
    matching for multi-value fields.

    Writes results to data/results/stage2_anls.csv.
    """
    from anls_star import anls_score

    raw_s2_path = output_dir / "stage2_raw_predictions.json"
    if not raw_s2_path.exists():
        print("  WARNING: stage2_raw_predictions.json not found; run stage 2 first.")
        return

    records   = _load_json(STAGE2_ANN)
    gt_map    = {r["email_id"]: r for r in records}
    raw_preds = json.load(open(raw_s2_path, encoding="utf-8"))

    approaches_present = sorted(set(e["approach"] for e in raw_preds))
    print(f"\n=== ANLS computation ===")
    print(f"  Approaches: {approaches_present}")

    # Group predictions by approach then email_id
    by_approach: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for entry in raw_preds:
        by_approach[entry["approach"]][entry["email_id"]] = entry["predictions"]

    results: dict[str, dict[str, float]] = {}
    for approach, email_preds in by_approach.items():
        field_scores: dict[str, list[float]] = defaultdict(list)

        for email_id, preds in email_preds.items():
            if email_id not in gt_map:
                continue
            gt   = _gt_anls_values(gt_map[email_id])
            pred = _pred_anls_values(preds)

            LIST_FIELDS = {"CC", "ATTACHMENT"}
            for field in EVAL_FIELDS:
                gt_val   = gt[field]
                pred_val = pred.get(field, [] if field in LIST_FIELDS else None)
                score = anls_score(gt_val, pred_val)
                field_scores[field].append(score)

        approach_result: dict[str, float] = {}
        all_scores: list[float] = []
        for field in EVAL_FIELDS:
            scores = field_scores[field]
            avg = round(sum(scores) / len(scores), 4) if scores else 0.0
            approach_result[field] = avg
            all_scores.extend(scores)
        approach_result["macro_anls"] = round(
            sum(approach_result[f] for f in EVAL_FIELDS) / len(EVAL_FIELDS), 4
        )
        approach_result["overall_anls"] = round(
            sum(all_scores) / len(all_scores), 4
        ) if all_scores else 0.0
        results[approach] = approach_result

    # Print table
    header = f"{'Approach':<25} {'Macro ANLS':>12} " + "  ".join(f"{f[:6]:>8}" for f in EVAL_FIELDS)
    print(header)
    print("-" * len(header))
    for approach in _STAGE2_ORDER:
        if approach not in results:
            continue
        r = results[approach]
        row = f"{approach:<25} {r['macro_anls']:>12.4f}  "
        row += "  ".join(f"{r[f]:>8.4f}" for f in EVAL_FIELDS)
        print(row)

    # Write CSV
    out_path = output_dir / "stage2_anls.csv"
    fieldnames = ["approach", "macro_anls", "overall_anls"] + EVAL_FIELDS
    rows = []
    for approach in _STAGE2_ORDER:
        if approach not in results:
            continue
        r = results[approach]
        rows.append({"approach": approach, **{k: r[k] for k in fieldnames[1:]}})

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {out_path.relative_to(_ROOT)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Regex / BERT / GPT-5.5 on Woo dossier extraction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stages", nargs="+", choices=["1", "2"], default=["1", "2"],
        metavar="{1,2}", help="Which stages to evaluate (default: 1 2)",
    )
    parser.add_argument(
        "--bert-s1", metavar="PATH",
        help="Trained stage-1 BERT checkpoint dir (line-level classifier)",
    )
    parser.add_argument(
        "--bert-s2", metavar="PATH",
        help="Trained stage-2 BERT checkpoint dir (field extraction)",
    )
    parser.add_argument(
        "--gpt", action="store_true",
        help="Run GPT-5.5 vision calls (requires OPENAI_API_KEY — costs money)",
    )
    parser.add_argument(
        "--anls", action="store_true",
        help="Compute ANLS* scores after evaluation (requires anls_star package)",
    )
    parser.add_argument(
        "--output-dir", default=str(DATA_DIR / "results"), metavar="PATH",
        help="Directory to write CSVs (default: data/results/)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    run_gpt = args.gpt

    # Load BERT models if checkpoints are provided
    bert_s1 = bert_s1_tok = bert_s2 = bert_s2_tok = None

    if args.bert_s1:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print(f"Loading stage-1 BERT (line-level) from {args.bert_s1} ...")
        bert_s1_tok = AutoTokenizer.from_pretrained(args.bert_s1)
        bert_s1 = AutoModelForSequenceClassification.from_pretrained(args.bert_s1)
        bert_s1.eval()

    if args.bert_s2:
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        print(f"Loading stage-2 BERT from {args.bert_s2} ...")
        bert_s2_tok = AutoTokenizer.from_pretrained(args.bert_s2)
        bert_s2 = AutoModelForTokenClassification.from_pretrained(args.bert_s2)
        bert_s2.eval()

    # Load ground truth
    stage1_records = _load_json(STAGE1_ANN)
    stage2_records = _load_json(STAGE2_ANN)

    # Build stage1 index for locating emails within dossiers (GPT vision needs page positions)
    # Uses stage1_test annotations; for cross-dossier lookup load stage1_train too
    stage1_index: dict[str, dict[str, tuple[int, int]]] = {}
    for ann_path in [STAGE1_ANN, DATA_DIR / "annotations" / "stage1_train.json"]:
        if ann_path.exists():
            for r in _load_json(ann_path):
                stage1_index[r["dossier_id"]] = {
                    e["email_id"]: (e["start_char"], e["end_char"]) for e in r["emails"]
                }

    # Build page maps for GPT-5.5 vision (stage 2 only)
    page_maps: dict[str, dict] = {}
    if "2" in args.stages and run_gpt:
        dossier_ids = list({r["dossier_id"] for r in stage2_records})
        print("Building page maps for GPT-5.5 vision ...")
        page_maps = _build_page_maps(dossier_ids)

    if "1" in args.stages:
        run_stage1(stage1_records, bert_s1, bert_s1_tok, run_gpt, output_dir)

    if "2" in args.stages:
        run_stage2(stage2_records, stage1_index, page_maps, bert_s2, bert_s2_tok, run_gpt, output_dir)

    if args.anls:
        run_anls(output_dir)

    print(f"\nDone. Results written to {output_dir.relative_to(_ROOT)}/")


if __name__ == "__main__":
    main()
