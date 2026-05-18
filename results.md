# Results Summary

## Stage 1 — Email Boundary Detection

Evaluated on the stage-1 test split (2 dossiers). Metric: span IoU >= 0.5 precision/recall/F1. BERT uses the line-level classifier with the proximity filter (collapsing multiple predictions within the same header block into a single boundary).

| Approach          | Precision | Recall | F1    |
|-------------------|-----------|--------|-------|
| Regex             | 0.926     | 0.922  | 0.924 |
| BERT (line+filter)| 0.966     | 0.901  | 0.932 |
| GPT-5.5 (vision)  | 0.971     | 0.971  | 0.971 |

GPT-5.5 achieves the highest F1 and is the only approach with near-perfect recall. BERT with the proximity filter improves markedly over the raw line classifier and outperforms regex in precision. Regex has competitive recall but lower precision due to spurious boundaries in forwarded email chains.

## Stage 2 — Field Extraction

Evaluated on 171 emails from the stage-2 test split. Two metrics are reported: exact macro F1 (value-level exact string match after normalisation) and ANLS (Average Normalised Levenshtein Similarity, macro-averaged across fields using `anls_star`).

| Approach         | Exact Macro F1 | ANLS  |
|------------------|----------------|-------|
| Regex            | 0.881          | 0.957 |
| BERT             | 0.312          | 0.571 |
| GPT-5.5 (vision) | 0.752          | 0.943 |

Regex outperforms BERT on exact match due to BERT's failure on CC, SUBJECT, and ATTACHMENT fields. GPT-5.5 substantially outperforms BERT and approaches regex on exact match; its ANLS score nearly matches regex, indicating it extracts semantically correct values even when redaction or OCR artefacts prevent exact matches.

## Stage 2 Per-Field Results

Exact F1 and ANLS (macro per-field ANLS from `anls_star`) for each field and approach:

| Field      | Regex F1 | Regex ANLS | BERT F1 | BERT ANLS | GPT-5.5 F1 | GPT-5.5 ANLS |
|------------|----------|------------|---------|-----------|------------|--------------|
| FROM       | 0.991    | 0.992      | 0.616   | 0.673     | 0.976      | 0.974        |
| TO         | 0.908    | 0.936      | 0.490   | 0.388     | 0.923      | 0.942        |
| CC         | 0.876    | 0.935      | 0.000   | 0.530     | 0.381      | 0.791        |
| DATE       | 0.977    | 0.974      | 0.769   | 0.937     | 0.971      | 0.992        |
| SUBJECT    | 0.909    | 0.968      | 0.000   | 0.000     | 0.906      | 0.990        |
| ATTACHMENT | 0.625    | 0.957      | 0.000   | 0.897     | 0.353      | 0.967        |

Notes:
- BERT F1 values are exact-match (value-level) micro P/R/F1 per field.
- ANLS values use the `anls_star` package (Peer et al., 2024) implementing the metric of Biten et al. (ICCV 2019), with Hungarian matching for multi-value fields (CC, ATTACHMENT).
- GPT-5.5 always uses prompt version 3 (vision + extracted text with explicit redaction handling instructions).
- BERT struggles with CC, SUBJECT, and ATTACHMENT because the training set contains relatively few examples of these fields; it achieves reasonable DATE extraction.
- Regex's ATTACHMENT ANLS (0.957) is high despite low F1 (0.625) because missed attachments score 0 on exact match but partial credit is given when filenames partially match.
