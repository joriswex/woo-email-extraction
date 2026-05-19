"""
Three-way evaluation: Regex vs. GPT-4o vs. BERT (optional)

Stage 1 — email boundary detection (dossier level)
  Approaches : Regex, GPT-4o (text), BERT (optional)
  Metric     : span IoU >= 0.5  →  precision / recall / F1
  Output     : data/results/stage1_per_dossier.csv
               data/results/stage1_summary.csv

Stage 2 — field extraction (email level, 60 dev emails)
  Approaches : Regex, GPT-4o (vision), BERT (optional)
  Metric     : value-level exact and fuzzy match (difflib >= 0.8)  →  P/R/F1
  Output     : data/results/stage2_per_field.csv
               data/results/stage2_summary.csv

Usage
-----
    # Regex only (default — free, no API key needed)
    python scripts/evaluate.py

    # Add GPT-4o (costs money — requires OPENAI_API_KEY)
    python scripts/evaluate.py --gpt

    # Full comparison: Regex + GPT-4o + BERT
    python scripts/evaluate.py --gpt --bert-s1 models/stage1_v1 --bert-s2 models/stage2_v1

    # Stage 2 only
    python scripts/evaluate.py --stages 2
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
STAGE1_ANN      = DATA_DIR / "annotations" / "stage1_test.json"
STAGE1_FULL_ANN = DATA_DIR / "annotations" / "stage1.json"  # all 8 dossiers — used for GPT-4o cross-reference only
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

_STAGE1_ORDER = ["Regex", "BERT", "BERT (line+filter)", "GPT-5.5 (text)", "GPT-5.5 (vision)"]
_STAGE2_ORDER = ["Regex", "BERT", "BERT (weighted)", "BERT (CRF v2)", "GPT-5.5 v3 (text)", "GPT-5.5 (vision)", "GPT-5.5 v2 (vision)", "GPT-5.5 v3 (vision)"]


def _merge_write_csv(
    path: Path,
    new_rows: list[dict],
    fieldnames: list[str],
    run_approaches: list[str],
    canonical_order: list[str],
) -> None:
    """Write CSV rows, preserving scores for approaches not run this time."""
    # Keep existing rows whose approach was NOT run this time
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
    print(f"  → {path.relative_to(_ROOT)}")


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
    # Load cache if it exists
    cache: dict[str, dict] = {}
    if _PAGE_MAP_CACHE.exists():
        with open(_PAGE_MAP_CACHE) as f:
            raw = json.load(f)
        # JSON keys are strings; page numbers need to be ints
        cache = {did: {int(k): tuple(v) for k, v in pm.items()} for did, pm in raw.items()}

    missing = [did for did in dossier_ids if did not in cache]
    if missing:
        from pdf_extract import extract_text
        print(f"  Building page maps for {len(missing)} dossier(s) (cached afterwards) …")
        for did in missing:
            pdf = RAW_DIR / f"{did}.pdf"
            if not pdf.exists():
                print(f"  WARNING: {did}.pdf not found, GPT-4o vision will be skipped")
                cache[did] = {}
                continue
            _, page_map = extract_text(pdf)
            cache[did] = page_map

        # Save updated cache
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
    bert_s1_arch: str = "token",
    bert_s1=None,
    bert_s1_tok=None,
    run_gpt: bool = True,
    run_gpt_vision: bool = False,
    page_maps: dict | None = None,
    output_dir: Path = DATA_DIR / "results",
) -> None:
    print("\n=== Stage 1: email boundary detection ===")

    # approach name → list of per-dossier metric dicts
    per_dossier: list[dict] = []
    approach_metrics: dict[str, list[dict]] = defaultdict(list)
    raw_preds_s1: dict = {}  # {dossier_id: {approach: [(start, end)]}} — saved for post-processing

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

        # BERT
        if bert_s1:
            if bert_s1_arch == "line":
                from bert_s1_predict import predict
            else:
                from bert_s1_predict import predict
            b_spans = predict(text, bert_s1, bert_s1_tok)
            b_m = compute_span_metrics(b_spans, gt_spans)
            per_dossier.append({"approach": "BERT", "dossier_id": did, **b_m})
            approach_metrics["BERT"].append(b_m)
            raw_preds_s1[did]["BERT"] = b_spans

        # GPT-5.5 text
        if run_gpt:
            import gpt_baseline
            try:
                g_spans = gpt_baseline.detect_email_boundaries_text(text)
                g_m = compute_span_metrics(g_spans, gt_spans)
                per_dossier.append({"approach": "GPT-5.5 (text)", "dossier_id": did, **g_m})
                approach_metrics["GPT-5.5 (text)"].append(g_m)
                raw_preds_s1[did]["GPT-5.5 (text)"] = g_spans
            except Exception as exc:
                print(f"    GPT-5.5 text error on {did}: {exc}")
                per_dossier.append({
                    "approach": "GPT-5.5 (text)", "dossier_id": did,
                    "precision": DASH, "recall": DASH, "f1": DASH,
                    "n_pred": DASH, "n_true": len(gt_spans),
                })

        # GPT-5.5 vision (images only — forces visual processing)
        if run_gpt_vision and page_maps is not None:
            import gpt_baseline
            pdf = RAW_DIR / f"{did}.pdf"
            pm  = page_maps.get(did, {})
            if pdf.exists() and pm:
                try:
                    gv_spans = gpt_baseline.detect_email_boundaries_vision(pdf, pm, text)
                    gv_m = compute_span_metrics(gv_spans, gt_spans)
                    per_dossier.append({"approach": "GPT-5.5 (vision)", "dossier_id": did, **gv_m})
                    approach_metrics["GPT-5.5 (vision)"].append(gv_m)
                    raw_preds_s1[did]["GPT-5.5 (vision)"] = gv_spans
                except Exception as exc:
                    print(f"    GPT-5.5 vision error on {did}: {exc}")
                    per_dossier.append({
                        "approach": "GPT-5.5 (vision)", "dossier_id": did,
                        "precision": DASH, "recall": DASH, "f1": DASH,
                        "n_pred": DASH, "n_true": len(gt_spans),
                    })
            else:
                print(f"    WARNING: PDF or page map missing for {did}, skipping vision")

    # Save raw predictions — merge with existing so previous approaches are preserved
    raw_s1_path = output_dir / "stage1_raw_predictions.json"
    raw_s1_path.parent.mkdir(parents=True, exist_ok=True)
    existing_s1: dict = {}
    if raw_s1_path.exists():
        with open(raw_s1_path) as f:
            existing_s1 = json.load(f)
    for did, apps in raw_preds_s1.items():
        if did not in existing_s1:
            existing_s1[did] = {}
        for approach, spans in apps.items():
            existing_s1[did][approach] = [list(s) for s in spans]
    with open(raw_s1_path, "w") as f:
        json.dump(existing_s1, f)
    print(f"  → {raw_s1_path.relative_to(_ROOT)}")

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
    stage1_index: dict,         # {dossier_id: {email_id: (start, end)}}
    page_maps: dict[str, dict], # {dossier_id: page_map}
    bert_s2_arch: str = "token",
    bert_s2=None,
    bert_s2_tok=None,
    run_gpt: bool = True,
    gpt_versions: list[int] = [1, 2, 3],
    run_gpt_text: bool = False,
    output_dir: Path = DATA_DIR / "results",
    bert_label: str | None = None,
) -> None:
    print(f"\n=== Stage 2: field extraction ({len(records)} emails) ===")

    # Accumulate micro-average counts per approach
    _GPT_LABELS = {1: "GPT-5.5 (vision)", 2: "GPT-5.5 v2 (vision)", 3: "GPT-5.5 v3 (vision)"}
    approaches = ["Regex"]
    if run_gpt:
        for v in gpt_versions:
            approaches.append(_GPT_LABELS[v])
    if run_gpt_text:
        approaches.append("GPT-5.5 v3 (text)")
    if bert_label is None:
        bert_label = "BERT (CRF v2)" if bert_s2_arch == "crf" else "BERT"
    if bert_s2:
        approaches.append(bert_label)

    counters: dict[str, dict] = {a: empty_counts() for a in approaches}
    raw_preds_s2: list[dict] = []  # [{dossier_id, email_id, approach, predictions}]

    for i, record in enumerate(records):
        did   = record["dossier_id"]
        eid   = record["email_id"]
        text  = record["text"]
        gt    = _gt_values(record)

        # Regex
        r_preds = [{"label": p["label"], "value": p["value"]}
                   for p in regex_fields(text) if p["label"] in EVAL_FIELDS]
        accumulate_field_counts(counters["Regex"], r_preds, gt, EVAL_FIELDS)
        raw_preds_s2.append({"dossier_id": did, "email_id": eid,
                              "approach": "Regex", "predictions": r_preds})

        # BERT
        if bert_s2:
            if bert_s2_arch == "crf":
                from bert_s2_predict import predict_fields_crf as predict_fields
            else:
                from bert_s2_predict import predict_fields
            b_spans = predict_fields(text, bert_s2, bert_s2_tok)
            b_preds = _spans_to_values(b_spans, text)
            accumulate_field_counts(counters[bert_label], b_preds, gt, EVAL_FIELDS)
            raw_preds_s2.append({"dossier_id": did, "email_id": eid,
                                  "approach": bert_label, "predictions": b_preds})

        # GPT-5.5 vision (v1 and v2)
        if run_gpt:
            pos = stage1_index.get(did, {}).get(eid)
            pm  = page_maps.get(did, {})
            if pos and pm:
                import gpt_baseline
                for v in gpt_versions:
                    label = _GPT_LABELS[v]
                    try:
                        g_preds = gpt_baseline.extract_fields_vision(
                            RAW_DIR / f"{did}.pdf", pm, pos[0], pos[1],
                            email_text=text, prompt_version=v,
                        )
                        accumulate_field_counts(counters[label], g_preds, gt, EVAL_FIELDS)
                        raw_preds_s2.append({"dossier_id": did, "email_id": eid,
                                             "approach": label, "predictions": g_preds})
                    except Exception as exc:
                        print(f"    GPT-5.5 v{v} error on {did}/{eid}: {exc}")
                    time.sleep(0.3)
            else:
                print(f"    WARNING: no position found for {did}/{eid}, skipping GPT-5.5")

        # GPT-5.5 text-only (no images — baseline for vision comparison)
        if run_gpt_text:
            import gpt_baseline
            try:
                gt_preds = gpt_baseline.extract_fields_text_only(text)
                accumulate_field_counts(counters["GPT-5.5 v3 (text)"], gt_preds, gt, EVAL_FIELDS)
                raw_preds_s2.append({"dossier_id": did, "email_id": eid,
                                     "approach": "GPT-5.5 v3 (text)", "predictions": gt_preds})
            except Exception as exc:
                print(f"    GPT-5.5 text-only error on {did}/{eid}: {exc}")
            time.sleep(0.3)

        if (i + 1) % 10 == 0 or (i + 1) == len(records):
            print(f"  {i + 1}/{len(records)} emails processed")

    run_approaches = list(counters.keys())

    # Save raw predictions — merge with existing file so previous approaches are preserved
    raw_s2_path = output_dir / "stage2_raw_predictions.json"
    existing_raw: list[dict] = []
    if raw_s2_path.exists():
        with open(raw_s2_path) as f:
            existing_raw = json.load(f)
    # Keep entries for approaches not run this time
    kept = [e for e in existing_raw if e["approach"] not in run_approaches]
    merged_raw = kept + raw_preds_s2
    with open(raw_s2_path, "w") as f:
        json.dump(merged_raw, f, ensure_ascii=False)
    print(f"  → {raw_s2_path.relative_to(_ROOT)}")

    # Build per-field CSV rows
    per_field_rows: list[dict] = []
    summary_rows: list[dict] = []

    for approach in approaches:
        field_metrics = prf_from_counts(counters[approach])
        for field in EVAL_FIELDS:
            if field in field_metrics:
                m = field_metrics[field]
                per_field_rows.append({
                    "approach": approach,
                    "field": field,
                    "exact_p":  m["exact_p"],
                    "exact_r":  m["exact_r"],
                    "exact_f1": m["exact_f1"],
                })
        mf = macro_f1(field_metrics)
        summary_rows.append({
            "approach": approach,
            "exact_macro_f1": mf["exact_f1"],
            "n_emails": len(records),
        })

    _merge_write_csv(
        output_dir / "stage2_per_field.csv",
        per_field_rows,
        ["approach", "field", "exact_p", "exact_r", "exact_f1"],
        run_approaches=run_approaches,
        canonical_order=_STAGE2_ORDER,
    )
    _merge_write_csv(
        output_dir / "stage2_summary.csv",
        summary_rows,
        ["approach", "exact_macro_f1", "n_emails"],
        run_approaches=run_approaches,
        canonical_order=_STAGE2_ORDER,
    )
    print("  Note: ANLS* scores computed separately via scripts/compute_anls.py")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Regex / GPT-4o / BERT on Woo dossier extraction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stages", nargs="+", choices=["1", "2"], default=["1", "2"],
        metavar="{1,2}", help="Which stages to evaluate (default: 1 2)",
    )
    parser.add_argument(
        "--bert-s1", metavar="PATH",
        help="Trained stage-1 BERT checkpoint dir (optional)",
    )
    parser.add_argument(
        "--bert-s1-arch", choices=["token", "line"], default="token",
        help="Stage-1 BERT architecture: 'token' (BIO, default) or 'line' (line-level classifier)",
    )
    parser.add_argument(
        "--bert-s2", metavar="PATH",
        help="Trained stage-2 BERT checkpoint dir (optional)",
    )
    parser.add_argument(
        "--bert-s2-label", metavar="LABEL", default=None,
        help="Row label for this BERT run in the CSV (default: 'BERT' or 'BERT (CRF v2)')",
    )
    parser.add_argument(
        "--bert-s2-arch", choices=["token", "crf"], default="token",
        help="Stage-2 BERT architecture: 'token' (BIO, default) or 'crf' (BERT+CRF)",
    )
    parser.add_argument(
        "--gpt", action="store_true",
        help="Run GPT vision+text calls for Stage 2 (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--gpt-versions", nargs="+", type=int, choices=[1, 2, 3], default=[1, 2, 3],
        metavar="{1,2,3}", help="Which GPT Stage-2 vision prompt versions to run (default: 1 2 3)",
    )
    parser.add_argument(
        "--gpt-s1-vision", action="store_true",
        help="Run GPT-5.5 vision (images only) for Stage 1 boundary detection",
    )
    parser.add_argument(
        "--gpt-s2-text", action="store_true",
        help="Run GPT-5.5 text-only for Stage 2 field extraction (no images)",
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
    bert_s1_arch = args.bert_s1_arch
    bert_s2_arch = args.bert_s2_arch

    if args.bert_s1:
        from transformers import AutoTokenizer
        bert_s1_tok = AutoTokenizer.from_pretrained(args.bert_s1)
        if bert_s1_arch == "line":
            from transformers import AutoModelForSequenceClassification
            print(f"Loading stage-1 BERT (line-level) from {args.bert_s1} …")
            bert_s1 = AutoModelForSequenceClassification.from_pretrained(args.bert_s1)
        else:
            from transformers import AutoModelForTokenClassification
            print(f"Loading stage-1 BERT (token) from {args.bert_s1} …")
            bert_s1 = AutoModelForTokenClassification.from_pretrained(args.bert_s1)
        bert_s1.eval()

    if args.bert_s2:
        from transformers import AutoTokenizer
        bert_s2_tok = AutoTokenizer.from_pretrained(args.bert_s2)
        if bert_s2_arch == "crf":
            import json as _json, torch as _torch
            from transformers import AutoModel
            from bert_s2_train import BertCRFForTokenClassification, NUM_LABELS
            print(f"Loading stage-2 BERT+CRF from {args.bert_s2} …")
            cfg = _json.loads((Path(args.bert_s2) / "crf_config.json").read_text())
            encoder = AutoModel.from_pretrained(cfg["base_model"])
            bert_s2 = BertCRFForTokenClassification(encoder, cfg["num_labels"])
            bert_s2.load_state_dict(_torch.load(Path(args.bert_s2) / "model_crf.pt", map_location="cpu"))
        else:
            from transformers import AutoModelForTokenClassification
            print(f"Loading stage-2 BERT (token) from {args.bert_s2} …")
            bert_s2 = AutoModelForTokenClassification.from_pretrained(args.bert_s2)
        bert_s2.eval()

    default_label = "BERT (CRF v2)" if bert_s2_arch == "crf" else "BERT"
    bert_label = args.bert_s2_label if args.bert_s2_label else default_label

    # Load ground truth
    stage1_records = _load_json(STAGE1_ANN)       # test split only (2 dossiers)
    stage2_records = _load_json(STAGE2_ANN)

    # Cross-reference index uses all 8 dossiers so GPT-4o vision can locate
    # any stage-2 email regardless of which dossier it came from
    stage1_index = {
        r["dossier_id"]: {
            e["email_id"]: (e["start_char"], e["end_char"]) for e in r["emails"]
        }
        for r in _load_json(STAGE1_FULL_ANN)
    }

    run_gpt_s1_vision = args.gpt_s1_vision
    run_gpt_s2_text   = args.gpt_s2_text

    # Build page maps for any vision calls
    page_maps: dict[str, dict] = {}
    need_page_maps = (
        ("2" in args.stages and run_gpt) or
        ("1" in args.stages and run_gpt_s1_vision)
    )
    if need_page_maps:
        s1_dossier_ids = [r["dossier_id"] for r in stage1_records]
        s2_dossier_ids = list({r["dossier_id"] for r in stage2_records})
        all_dossier_ids = list({*s1_dossier_ids, *s2_dossier_ids})
        print("Building page maps for vision calls …")
        page_maps = _build_page_maps(all_dossier_ids)

    if "1" in args.stages:
        run_stage1(stage1_records, bert_s1_arch, bert_s1, bert_s1_tok,
                   run_gpt, run_gpt_s1_vision, page_maps, output_dir)

    if "2" in args.stages:
        run_stage2(stage2_records, stage1_index, page_maps, bert_s2_arch, bert_s2, bert_s2_tok,
                   run_gpt, args.gpt_versions, run_gpt_s2_text, output_dir, bert_label)

    print(f"\nDone. Results written to {output_dir.relative_to(_ROOT)}/")


if __name__ == "__main__":
    main()
