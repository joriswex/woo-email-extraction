"""
Compute inter-rater agreement (IRA) for Stage 1 and Stage 2 annotations.

Metrics
-------
Stage 1 (email boundary detection):
  - Cohen's κ at line level  : each line labelled 1 (email start) or 0 (not)
  - Boundary F1              : treat annotator A as ground truth, B as prediction

Stage 2 (field extraction):
  - Cohen's κ at word level  : each whitespace token assigned its field label or O
  - Per-field span F1        : exact span match (same start, end, label)
  - Macro-average F1         : unweighted mean across all field types

Interpretation (Landis & Koch, 1977):
  κ > 0.80  — almost perfect agreement   (excellent)
  κ 0.61–0.80 — substantial agreement    (acceptable for complex NLP tasks)
  κ 0.41–0.60 — moderate agreement       (requires discussion)
  κ < 0.41  — fair/slight agreement      (problematic)

Usage
-----
    python scripts/ira/compute_ira.py \\
        --stage1-a data/ira/exports/stage1_rater_a.json \\
        --stage1-b data/ira/exports/stage1_rater_b.json \\
        --stage2-a data/ira/exports/stage2_rater_a.json \\
        --stage2-b data/ira/exports/stage2_rater_b.json

    # Stage 2 only (if stage 1 exports not yet available):
    python scripts/ira/compute_ira.py \\
        --stage2-a data/ira/exports/stage2_rater_a.json \\
        --stage2-b data/ira/exports/stage2_rater_b.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import cohen_kappa_score


# ---------------------------------------------------------------------------
# Label Studio export helpers
# ---------------------------------------------------------------------------


def _best_annotation(task: dict) -> list[dict]:
    """Return the result list of the first non-cancelled annotation."""
    for ann in task.get("annotations", []):
        if not ann.get("was_cancelled", False):
            return ann.get("result", [])
    return []


def _spans_from_result(result: list[dict]) -> list[tuple[int, int, str]]:
    """Extract (start, end, label) tuples from a Label Studio result list."""
    spans = []
    for r in result:
        v = r.get("value", {})
        labels = v.get("labels", [])
        if labels and "start" in v and "end" in v:
            spans.append((v["start"], v["end"], labels[0]))
    return spans


# ---------------------------------------------------------------------------
# Word-level label assignment (used for Cohen's κ)
# ---------------------------------------------------------------------------


def _word_labels(text: str, spans: list[tuple[int, int, str]]) -> list[str]:
    """
    Assign a label to each whitespace-delimited token in text.

    A token gets the label of the first span it overlaps at all (outward
    rounding), otherwise O. This mirrors bert_s2_data._find_overlapping_field,
    which is what the model is actually trained on.
    """
    labels = []
    for match in re.finditer(r"\S+", text):
        t_start, t_end = match.start(), match.end()
        label = "O"
        for s_start, s_end, s_label in spans:
            if t_start < s_end and t_end > s_start:
                label = s_label
                break
        labels.append(label)
    return labels


# ---------------------------------------------------------------------------
# Stage 1 IRA
# ---------------------------------------------------------------------------


def _stage1_boundary_sets(export: list[dict]) -> dict[str, set[int]]:
    """Return {dossier_id: set of email start char positions} from a Stage 1 export."""
    result: dict[str, set[int]] = {}
    for task in export:
        did = task["data"].get("dossier_id", task["data"].get("text", "")[:10])
        anns = _best_annotation(task)
        starts = set()
        for r in anns:
            v = r.get("value", {})
            if v.get("labels") == ["EMAIL"] and "start" in v:
                starts.add(v["start"])
        result[did] = starts
    return result


def _stage1_line_labels(text: str, starts: set[int]) -> list[int]:
    """Convert a set of email start char positions to a per-line binary label sequence."""
    labels = []
    pos = 0
    for line in text.split("\n"):
        if line.strip():  # skip blank lines
            labels.append(1 if pos in starts else 0)
        pos += len(line) + 1
    return labels


def compute_stage1_ira(path_a: Path, path_b: Path) -> None:
    export_a = json.load(open(path_a, encoding="utf-8"))
    export_b = json.load(open(path_b, encoding="utf-8"))

    # Build dossier text lookup from rater A (both have the same texts)
    texts: dict[str, str] = {
        t["data"]["dossier_id"]: t["data"]["text"] for t in export_a
    }

    sets_a = _stage1_boundary_sets(export_a)
    sets_b = _stage1_boundary_sets(export_b)

    print("=" * 60)
    print("STAGE 1 — Email Boundary Detection")
    print("=" * 60)

    all_labels_a: list[int] = []
    all_labels_b: list[int] = []
    all_p, all_r, all_f1 = [], [], []

    for did in sorted(sets_a):
        if did not in sets_b:
            print(f"  WARNING: {did} missing from rater B export — skipping")
            continue

        text = texts[did]
        sa, sb = sets_a[did], sets_b[did]

        # Boundary F1 (exact position match)
        tp = len(sa & sb)
        precision = tp / len(sa) if sa else 0.0
        recall    = tp / len(sb) if sb else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        all_p.append(precision); all_r.append(recall); all_f1.append(f1)

        # Line-level labels for κ
        la = _stage1_line_labels(text, sa)
        lb = _stage1_line_labels(text, sb)
        all_labels_a.extend(la)
        all_labels_b.extend(lb)

        print(f"  {did}: A={len(sa)} boundaries  B={len(sb)} boundaries  "
              f"P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}")

    print()
    kappa = cohen_kappa_score(all_labels_a, all_labels_b)
    macro_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0.0
    print(f"  Line-level Cohen's κ : {kappa:.4f}")
    print(f"  Boundary macro-F1    : {macro_f1:.4f}")
    print(f"  Boundary macro-P     : {sum(all_p)/len(all_p):.4f}")
    print(f"  Boundary macro-R     : {sum(all_r)/len(all_r):.4f}")
    _interpret_kappa(kappa)
    print()


# ---------------------------------------------------------------------------
# Stage 2 IRA
# ---------------------------------------------------------------------------


FIELD_LABELS = ["FROM", "TO", "CC", "DATE", "SUBJECT", "ATTACHMENT"]


def compute_stage2_ira(path_a: Path, path_b: Path) -> None:
    export_a = json.load(open(path_a, encoding="utf-8"))
    export_b = json.load(open(path_b, encoding="utf-8"))

    # Index rater B by (dossier_id, email_id) for fast lookup
    index_b: dict[tuple[str, str], list[dict]] = {}
    for task in export_b:
        key = (task["data"]["dossier_id"], task["data"]["email_id"])
        index_b[key] = _best_annotation(task)

    print("=" * 60)
    print("STAGE 2 — Field Extraction")
    print("=" * 60)

    all_labels_a: list[str] = []
    all_labels_b: list[str] = []

    # Per-field span counts for F1
    span_tp:  dict[str, int] = defaultdict(int)
    span_fa:  dict[str, int] = defaultdict(int)  # in A not in B (false positives of A)
    span_fb:  dict[str, int] = defaultdict(int)  # in B not in A (false positives of B)

    skipped = 0
    for task in export_a:
        did      = task["data"]["dossier_id"]
        eid      = task["data"]["email_id"]
        text     = task["data"]["text"]
        key      = (did, eid)

        if key not in index_b:
            skipped += 1
            continue

        spans_a = _spans_from_result(_best_annotation(task))
        spans_b = _spans_from_result(index_b[key])

        # Word-level labels for κ
        wa = _word_labels(text, spans_a)
        wb = _word_labels(text, spans_b)
        # Align: both lists derived from same text so length is equal
        all_labels_a.extend(wa)
        all_labels_b.extend(wb)

        # Per-field span F1
        set_a = set(spans_a)
        set_b = set(spans_b)
        for span in set_a & set_b:
            span_tp[span[2]] += 1
        for span in set_a - set_b:
            span_fa[span[2]] += 1
        for span in set_b - set_a:
            span_fb[span[2]] += 1

    if skipped:
        print(f"  WARNING: {skipped} tasks missing from rater B export — skipped\n")

    # Cohen's κ (word level)
    kappa = cohen_kappa_score(all_labels_a, all_labels_b)

    # Per-field F1
    print(f"  {'Field':<12}  {'TP':>5}  {'A only':>6}  {'B only':>6}  "
          f"{'P':>6}  {'R':>6}  {'F1':>6}")
    print("  " + "-" * 56)
    p_scores, r_scores, f1_scores = [], [], []
    for label in FIELD_LABELS:
        tp = span_tp[label]
        fa = span_fa[label]
        fb = span_fb[label]
        p  = tp / (tp + fa) if (tp + fa) > 0 else 0.0
        r  = tp / (tp + fb) if (tp + fb) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        p_scores.append(p); r_scores.append(r); f1_scores.append(f1)
        print(f"  {label:<12}  {tp:>5}  {fa:>6}  {fb:>6}  {p:>6.3f}  {r:>6.3f}  {f1:>6.3f}")

    macro_p  = sum(p_scores) / len(p_scores)
    macro_r  = sum(r_scores) / len(r_scores)
    macro_f1 = sum(f1_scores) / len(f1_scores)
    print("  " + "-" * 56)
    print(f"  {'Macro avg':<12}  {'':>5}  {'':>6}  {'':>6}  {macro_p:>6.3f}  {macro_r:>6.3f}  {macro_f1:>6.3f}")
    print()
    print(f"  Word-level Cohen's κ : {kappa:.4f}")
    print(f"  Span macro-P         : {macro_p:.4f}")
    print(f"  Span macro-R         : {macro_r:.4f}")
    print(f"  Span macro-F1        : {macro_f1:.4f}")
    _interpret_kappa(kappa)
    print()


# ---------------------------------------------------------------------------
# Interpretation helper
# ---------------------------------------------------------------------------


def _interpret_kappa(kappa: float) -> None:
    if kappa > 0.80:
        level = "almost perfect"
    elif kappa > 0.60:
        level = "substantial"
    elif kappa > 0.40:
        level = "moderate"
    elif kappa > 0.20:
        level = "fair"
    else:
        level = "slight / poor"
    print(f"  Interpretation       : {level} (Landis & Koch, 1977)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute inter-rater agreement for Stage 1 and/or Stage 2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stage1-a", metavar="PATH", help="Stage 1 export — rater A")
    parser.add_argument("--stage1-b", metavar="PATH", help="Stage 1 export — rater B")
    parser.add_argument("--stage2-a", metavar="PATH", help="Stage 2 export — rater A")
    parser.add_argument("--stage2-b", metavar="PATH", help="Stage 2 export — rater B")
    args = parser.parse_args()

    if not any([args.stage1_a, args.stage2_a]):
        parser.error("Provide at least --stage1-a/--stage1-b or --stage2-a/--stage2-b.")

    if args.stage1_a or args.stage1_b:
        if not (args.stage1_a and args.stage1_b):
            parser.error("Provide both --stage1-a and --stage1-b.")
        compute_stage1_ira(Path(args.stage1_a), Path(args.stage1_b))

    if args.stage2_a or args.stage2_b:
        if not (args.stage2_a and args.stage2_b):
            parser.error("Provide both --stage2-a and --stage2-b.")
        compute_stage2_ira(Path(args.stage2_a), Path(args.stage2_b))


if __name__ == "__main__":
    main()
