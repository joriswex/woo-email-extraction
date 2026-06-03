# Results Summary

All scores computed on held-out test sets using `scripts/evaluate.py` (exact F1)
and `scripts/compute_anls.py` (ANLS*). ANLS* uses the `anls-star` package
(Peer et al., 2024), restricted to emails where the field is present to avoid
inflating scores from correct non-predictions on absent fields.

---

## Stage 1 — Email Boundary Detection

Evaluated on 2 held-out test dossiers (242 ground-truth email boundaries).
A prediction is a true positive if IoU with the ground truth span ≥ 0.5.

| Approach | Precision | Recall | F1 |
|---|---|---|---|
| Regex | 0.926 | 0.922 | 0.924 |
| RobBERT (token BIO) | 0.370 | 0.943 | 0.531 |
| RobBERT (line + proximity filter) | **0.966** | 0.901 | 0.932 |
| GPT-5.5 (text) | **0.971** | **0.971** | **0.971** |
| GPT-5.5 (vision+text) | 0.913 | 0.656 | 0.726 |

---

## Stage 2 — Email Field Extraction

Evaluated on 171 held-out emails (stratified 20% split across all 8 dossiers).

### Macro-level summary

| Approach | Exact macro F1 | ANLS* (present-field) |
|---|---|---|
| Regex | **0.881** | **0.886** |
| RobBERT (token, unweighted) | 0.312 | 0.330 |
| RobBERT (token, class-weighted) | 0.465 | 0.858 |
| GPT-5.5 (text) | 0.741 | 0.832 |
| GPT-5.5 (vision+text) | **0.752** | 0.841 |

### Per-field breakdown

| Field | Regex F1 / ANLS* | RobBERT (token, unweighted) F1 / ANLS* | RobBERT (token, class-weighted) F1 / ANLS* | GPT-5.5 (text) F1 / ANLS* | GPT-5.5 (vision+text) F1 / ANLS* |
|---|---|---|---|---|---|
| FROM | 0.991 / 0.994 | 0.616 / 0.660 | 0.617 / 0.905 | 0.988 / 0.994 | 0.976 / 0.976 |
| TO | 0.908 / 0.942 | 0.489 / 0.390 | 0.539 / 0.845 | 0.939 / 0.955 | 0.923 / 0.940 |
| CC | 0.876 / 0.890 | 0.000 / 0.000 | 0.548 / 0.864 | 0.346 / 0.603 | 0.381 / 0.603 |
| DATE | 0.977 / 0.983 | 0.769 / 0.928 | 0.778 / 0.983 | 0.983 / 0.994 | 0.971 / 0.971 |
| SUBJECT | 0.909 / 0.954 | 0.000 / 0.000 | 0.306 / 0.886 | 0.925 / 0.988 | 0.906 / 0.964 |
| ATTACHMENT | 0.625 / 0.557 | 0.000 / 0.000 | 0.000 / 0.662 | 0.264 / 0.461 | 0.353 / 0.594 |

---

## Notes

- **RobBERT (token, unweighted)**: Focal Loss only, no class weights. Collapses to
  F1=0.000 on CC, SUBJECT, and ATTACHMENT due to severe token-level class
  imbalance (O tokens = 70% of training data).
- **RobBERT (token, class-weighted)**: Same architecture with sqrt-inverse-frequency class
  weights as Focal Loss α. Breaks label collapse; CC recovers to F1=0.548.
- **GPT-5.5 text**: Stage 1 uses full dossier text with date-line anchoring.
  Stage 2 uses extracted email text only (no images), prompt v3.
- **GPT-5.5 vision+text**: Stage 1 uses all PDF pages at 72 DPI + dossier text.
  Stage 2 uses email PDF pages at 150 DPI + email text, prompt v3.
- ANLS* computed via `anls-star` v1.0.1. See `scripts/compute_anls.py`.
