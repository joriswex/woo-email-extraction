"""
Evaluation metrics shared across all three approaches.

All approaches are compared at the value level (extracted text), not span level.
BERT and regex span predictions are converted to text via email_text[start:end];
GPT returns text directly. This is the only fair comparison across all three.

Two match modes:
  exact  — normalised strings must be equal (lowercased, collapsed whitespace)
  anls   — Average Normalised Levenshtein Similarity >= 0.5 threshold
           (Biten et al., ICCV 2019; standard metric for OCR document understanding)
           NLS = 1 - edit_distance(a, b) / max(len(a), len(b))
           Counts as match if NLS >= ANLS_THRESHOLD (0.5), else 0.

Aggregation:
  compute_field_counts()  — accumulate TP / n_pred / n_gt per field per record
  prf_from_counts()       — compute P/R/F1 from accumulated counts (micro-average)
  compute_span_metrics()  — IoU-based P/R/F1 for email boundary detection (stage 1)
"""
from __future__ import annotations

from collections import defaultdict

ANLS_THRESHOLD = 0.5
SPAN_IOU_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Internal helpers (also used by evaluate.py)
# ---------------------------------------------------------------------------


def _norm(s: str) -> str:
    return " ".join(s.strip().split()).lower()


def _levenshtein(a: str, b: str) -> int:
    """Standard dynamic-programming Levenshtein edit distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + (ca != cb),  # substitution
            )
        prev = curr
    return prev[len(b)]


def _anls_score(a: str, b: str) -> float:
    """Normalised Levenshtein similarity in [0, 1]."""
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - _levenshtein(a, b) / max_len


def exact_match(a: str, b: str) -> bool:
    return _norm(a) == _norm(b)


def anls_match(a: str, b: str) -> bool:
    """True if ANLS score >= threshold (Biten et al., ICCV 2019)."""
    return _anls_score(_norm(a), _norm(b)) >= ANLS_THRESHOLD


def _greedy_tp(preds: list[str], gts: list[str], match_fn) -> int:
    """Greedy 1-to-1 matching; returns number of true positives."""
    matched = set()
    tp = 0
    for pv in preds:
        for i, gv in enumerate(gts):
            if i not in matched and match_fn(pv, gv):
                tp += 1
                matched.add(i)
                break
    return tp


# ---------------------------------------------------------------------------
# Public: field-level accumulator (micro-average across records)
# ---------------------------------------------------------------------------


def empty_counts() -> dict:
    """Return a fresh counter dict for one approach."""
    return defaultdict(lambda: {"exact_tp": 0, "anls_tp": 0, "n_pred": 0, "n_gt": 0})


def accumulate_field_counts(
    counters: dict,
    predictions: list[dict],    # [{"label": str, "value": str}]
    ground_truths: list[dict],  # [{"label": str, "value": str}]
    eval_fields: list[str],
) -> None:
    """
    Update counters in-place for one record.

    Call once per record; after all records call prf_from_counts().
    """
    from collections import defaultdict as _dd
    pred_by = _dd(list)
    gt_by = _dd(list)
    for p in predictions:
        pred_by[p["label"]].append(p["value"])
    for g in ground_truths:
        gt_by[g["label"]].append(g["value"])

    for field in eval_fields:
        fp = pred_by.get(field, [])
        fg = gt_by.get(field, [])
        counters[field]["n_pred"] += len(fp)
        counters[field]["n_gt"] += len(fg)
        counters[field]["exact_tp"] += _greedy_tp(fp, fg, exact_match)
        counters[field]["anls_tp"]  += _greedy_tp(fp, fg, anls_match)


def prf_from_counts(counters: dict) -> dict[str, dict]:
    """
    Compute precision / recall / F1 from accumulated counts.

    Returns {"FIELD": {"exact_p": float, "exact_r": float, "exact_f1": float,
                        "fuzzy_p": float, "fuzzy_r": float, "fuzzy_f1": float}}
    """
    def _prf(tp, np_, ng):
        p = tp / np_ if np_ else 0.0
        r = tp / ng if ng else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return round(p, 4), round(r, 4), round(f, 4)

    results = {}
    for field, c in counters.items():
        row = {}
        for prefix, tp_key in [("exact", "exact_tp"), ("anls", "anls_tp")]:
            p, r, f = _prf(c[tp_key], c["n_pred"], c["n_gt"])
            row[f"{prefix}_p"]  = p
            row[f"{prefix}_r"]  = r
            row[f"{prefix}_f1"] = f
        results[field] = row
    return results


def macro_f1(per_field: dict[str, dict]) -> dict[str, float]:
    """Macro-average F1 across all field types."""
    if not per_field:
        return {"exact_f1": 0.0, "fuzzy_f1": 0.0}
    vals = list(per_field.values())
    return {
        "exact_f1": round(sum(v["exact_f1"] for v in vals) / len(vals), 4),
        "anls_f1":  round(sum(v["anls_f1"]  for v in vals) / len(vals), 4),
    }


# ---------------------------------------------------------------------------
# Public: span-level metrics for email boundary detection (stage 1)
# ---------------------------------------------------------------------------


def compute_span_metrics(
    predicted: list[tuple[int, int]],
    true: list[tuple[int, int]],
    iou_threshold: float = SPAN_IOU_THRESHOLD,
) -> dict[str, float]:
    """
    Span P/R/F1 where a prediction is TP if IoU with any true span >= threshold.
    """
    def iou(a, b):
        inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
        union = (a[1] - a[0]) + (b[1] - b[0]) - inter
        return inter / union if union > 0 else 0.0

    matched = set()
    tp = 0
    for ps in predicted:
        for i, ts in enumerate(true):
            if i not in matched and iou(ps, ts) >= iou_threshold:
                tp += 1
                matched.add(i)
                break

    n_pred, n_true = len(predicted), len(true)
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_true if n_true else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f, 4),
        "n_pred": n_pred,
        "n_true": n_true,
    }
