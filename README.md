# Reading between the ████████: Reconstructing Emails from Redacted Dutch FOIA Disclosures

This repository contains the code and data for a master's thesis in the **MSc Cultural Data and AI**
programme at the **University of Amsterdam** (Joris Wechsler, 2026).

The central research question is: *To what extent can machine learning methods help to reconstruct
emails from Dutch Woo-dossiers into structured, navigable records, and how could structural
improvements in the release workflow of such documents facilitate FOIA usability and accessibility?*

The project compares three approaches to extracting structured metadata from Dutch government Woo
(*Wet open overheid*) dossier PDFs: a regex baseline, a fine-tuned RobBERT model, and GPT-5.5. The
task is split into two stages — email boundary detection (Stage 1) and email field extraction
(Stage 2) — evaluated on a manually annotated corpus of 2,013 emails across 20 Woo PDF dossiers.
Stage 1 uses a dossier-level holdout (10 train / 10 test dossiers); Stage 2 uses an email-level
stratified holdout (1,006 train / 1,007 test emails across all 20 dossiers).

All coding was done in Python using VS Code, with coding assistance from Claude Code. BERT training
was performed on Kaggle cloud infrastructure (NVIDIA Tesla T4, 2×15 GB VRAM). Inference and
evaluation ran locally on an Apple M3 8-core CPU with 8 GB unified memory.

---

## Repository structure

```
Thesis_Project_Clean/
├── src/                                  # Core modules (add to sys.path)
│   ├── pdf_extract.py                    # PDF text extraction via PyMuPDF + Tesseract OCR
│   ├── regex_baseline.py                 # Regex pipeline: boundary detection and field extraction
│   ├── gpt_baseline.py                   # GPT-5.5 pipeline (text + vision modes, both stages)
│   ├── bert_s1_data.py                   # Stage 1 annotation schema and BIO label mapping
│   ├── bert_s1_train.py                  # Stage 1 BERT training (line-level classifier)
│   ├── bert_s1_predict.py                # Stage 1 BERT inference (line-level)
│   ├── bert_s2_data.py                   # Stage 2 field label schema and BIO label mapping
│   ├── bert_s2_train.py                  # Stage 2 BERT training (token classification)
│   ├── bert_s2_predict.py                # Stage 2 BERT inference
│   ├── bert_tokenize.py                  # Sliding-window tokenisation (stride=128, max=512)
│   ├── focal_loss.py                     # Focal Loss (Lin et al., 2017) with optional class weights
│   └── eval_metrics.py                   # Evaluation metrics: span IoU, exact match, ANLS*
├── scripts/
│   ├── evaluate.py                       # Main evaluation script — runs all approaches, writes CSVs
│   ├── metrics/                          # Post-hoc metric and analysis scripts
│   │   ├── compute_anls.py               # ANLS* metric from saved predictions (anls-star package)
│   │   ├── compute_iou_sweep.py          # Stage 1 IoU threshold sweep from saved predictions
│   │   ├── compute_approach_correlations.py  # Pairwise Pearson correlation of per-dossier scores
│   │   ├── compare_proximity_filter.py   # Post-hoc proximity filter analysis for RobBERT Stage 1
│   │   └── plot_dossier_variance.py      # Per-dossier bar charts for Stage 1 and Stage 2 (SVG)
│   ├── annotation/                       # Annotation workflow scripts (pre-annotation + Label Studio)
│   │   ├── prepare_import.py             # Prepare Label Studio import tasks from PDF
│   │   ├── preannotate.py                # Regex-based pre-annotation for Label Studio
│   │   ├── convert_annotations.py        # Convert Label Studio exports to train/test JSON splits
│   │   └── manage_dossiers.py            # Dossier management and overview utility
│   ├── ira/                              # Inter-rater agreement scripts
│   │   ├── create_ira_tasks.py           # Create IRA annotation tasks
│   │   └── compute_ira.py                # Compute Cohen's κ and macro F1 between raters
│   └── labelstudio/                      # Label Studio configuration and export conversion
│       ├── labeling_config_stage1.xml    # Label Studio config for Stage 1
│       ├── labeling_config_stage2.xml    # Label Studio config for Stage 2
│       ├── convert_labelstudio_export.py # Convert raw Label Studio exports
│       └── export_stage1_to_stage2.py    # Carry Stage 1 spans forward into Stage 2 tasks
├── notebooks/
│   ├── train_bert_stage1_kaggle.py       # Stage 1 training notebook (Kaggle, GPU required)
│   └── train_bert_stage2_kaggle.py       # Stage 2 training notebook — trains unweighted + weighted
├── data/
│   ├── annotations/                      # Ground-truth annotation files (train/test splits)
│   │   ├── stage1_train.json             # 10 dossiers, 1,006 emails (Stage 1 training)
│   │   ├── stage1_test.json              # 10 dossiers, 1,007 emails (Stage 1 test)
│   │   ├── stage2_train.json             # 1,006 emails stratified across 20 dossiers
│   │   └── stage2_test.json              # 1,007 emails stratified across 20 dossiers
│   ├── ira/                              # Inter-rater agreement data and annotation guide
│   │   ├── annotation_guide.md           # Guide given to secondary rater (also in thesis Appendix B)
│   │   ├── annotation_guide.docx         # Word version of the annotation guide
│   │   └── exports/                      # Raw annotation exports from both raters
│   └── results/                          # Evaluation outputs (CSVs, SVGs, raw predictions)
├── requirements.txt                      # Pinned Python dependencies
├── README.md
└── results.md                            # Full results summary with all tables
```

---

## Three pipelines

### Regex baseline (`src/regex_baseline.py`)
Pattern-based extraction using Dutch/English email header keywords (`Van:`, `From:`, `Aan:`,
`To:`, etc.). For Stage 1, a combination of a FROM/TO field and a SUBJECT field within the
next 500 characters triggers a new email boundary. For Stage 2, keywords are matched at
line starts and values captured as the text following the colon. Computationally inexpensive
and requires no training data.

### RobBERT (`src/bert_s1_train.py`, `src/bert_s2_train.py`)
Fine-tuned `DTAI-KULeuven/robbert-2023-dutch-base`.

**Stage 1** uses a line-level binary classifier (email start vs. not), with ±2 surrounding
context lines as input. This avoids the 512-token window over-segmentation problem of
token-level BIO labelling. Training uses Focal Loss (γ=2.0) with sqrt-inverse-frequency class
weights as α. The headline result (macro F1 = 0.769) is the unfiltered inference output; a
post-hoc proximity filter (`merge_nearby_starts`, max_lines=8) is analysed separately in
`scripts/metrics/compare_proximity_filter.py`.

**Stage 2** uses BIO token classification across 13 labels (B-/I- for each of 6 fields +
O). Two variants are trained: without class weights (showing label collapse on rare fields)
and with sqrt-inverse-frequency class weights as the Focal Loss α. Both use identical
hyperparameters (15-epoch ceiling, patience=5, batch size=16, seed=0 for dev split).

### GPT-5.5 (`src/gpt_baseline.py`)
Calls `gpt-5.5-2026-04-23` (pinned snapshot; reasoning effort set to API default). Both
text-only and vision+text modes are evaluated for each stage:

- **Stage 1 text**: full dossier text submitted as a single prompt; DATE field used as
  anchor to locate email boundaries
- **Stage 1 vision+text**: all PDF pages at 72 DPI + dossier text; model uses images as
  primary source, text as reference for copying date strings
- **Stage 2 text**: extracted email text only; prompt v3 handles redaction codes and OCR
  artefacts
- **Stage 2 vision+text**: email PDF pages at 150 DPI + email text; same prompt v3

---

## Dataset

Twenty Woo PDF dossiers were selected to represent variety of ministries, redaction
practices and email formats. After removing non-email pages, 1,248 pages of email data
remained (shortest: 12 pages, longest: 185 pages, average: 62.5 pages).

**Text extraction**: All dossiers were processed with PyMuPDF (page rendering at 200 DPI)
and Tesseract OCR 5.5.2 (`nld+eng`). Native pdfplumber extraction was evaluated first — six
dossiers produced readable output, but OCR artefacts were already present in those files.
Tesseract was used for all dossiers to ensure a consistent, reproducible text layer.

**Annotation**: Pre-annotations were generated using the regex pipeline and manually
verified and corrected in Label Studio in two rounds (Stage 1 first, then Stage 2). Of the
2,013 email boundary annotations, 39.6% were adjusted. Of the 9,077 field-level
annotations, 16.6% were adjusted.

**Train/test splits**:
- Stage 1: greedy email-count-balanced split at the dossier level → 10 train / 10 test
  dossiers (fully deterministic, no random seed)
- Stage 2: stratified 50/50 split within each dossier (seed=42) → 1,006 train / 1,007 test
  emails; all 20 dossiers present in both splits

**Inter-rater agreement**: Two dossiers were annotated by a secondary rater (no
pre-annotations provided). Cohen's κ = 0.970 (Stage 1, line-level); κ = 0.953 (Stage 2,
word-level). See `data/ira/` and `scripts/ira/compute_ira.py`.

Annotation files are stored as JSON arrays in `data/annotations/`. Character offsets are
relative to the Tesseract OCR-extracted text (`src/pdf_extract.py`).

---

## How to run evaluation

```bash
# Install dependencies
pip install -r requirements.txt

# Regex only (no API key needed — reproduces saved predictions)
python scripts/evaluate.py

# GPT-5.5 Stage 2 vision+text (requires OPENAI_API_KEY)
python scripts/evaluate.py --stages 2 --gpt --gpt-versions 3

# GPT-5.5 Stage 1 vision+text
python scripts/evaluate.py --stages 1 --gpt-s1-vision

# GPT-5.5 Stage 2 text-only
python scripts/evaluate.py --stages 2 --gpt-s2-text

# RobBERT Stage 2 (weighted model)
python scripts/evaluate.py --stages 2 --bert-s2 models/stage2_weighted_v2 \
    --bert-s2-label "RobBERT (token, class-weighted)"

# RobBERT Stage 1 (line classifier)
python scripts/evaluate.py --stages 1 --bert-s1 models/stage1_line_v2 \
    --bert-s1-arch line

# Compute ANLS* scores from saved raw predictions (run after evaluate.py)
python scripts/metrics/compute_anls.py

# IoU threshold sweep for Stage 1
python scripts/metrics/compute_iou_sweep.py

# Pairwise cross-approach Pearson correlations
python scripts/metrics/compute_approach_correlations.py

# Per-dossier variance bar charts (Stage 1 and Stage 2)
python scripts/metrics/plot_dossier_variance.py --stage 1
python scripts/metrics/plot_dossier_variance.py --stage 2
```

Results are written to `data/results/`. Raw predictions are cached in
`data/results/stage1_raw_predictions.json` and `data/results/stage2_raw_predictions.json`
so evaluation metrics can be recomputed without re-running inference.

---

## How to train BERT

Training notebooks in `notebooks/` are designed for Kaggle (GPU required). Upload the
relevant `src/` files and annotation JSON to a Kaggle dataset (`woolens-data`) before
running.

| Notebook | Result row(s) | Key settings |
|---|---|---|
| `train_bert_stage1_kaggle.py` | RobBERT (line, unfiltered) | 15-epoch ceiling, patience=3, batch=32, Focal Loss γ=2.0 + sqrt-inverse-freq weights |
| `train_bert_stage2_kaggle.py` | RobBERT (unweighted) + RobBERT (class-weighted) | 15-epoch ceiling, patience=5, batch=16, Focal Loss γ=2.0; `use_class_weights` flag controls the two variants |

Both notebooks use `random.Random(0)` to carve out a 10% dev set (Stage 1: 2 dossiers out
of 10; Stage 2: ~101 emails out of 1,006) for early stopping. The dev set is part of the
training corpus and is not the held-out test set.

Trained models were uploaded to HuggingFace (`joriswechs/woolens-stage2`) and downloaded
locally into `models/` before evaluation.

---

## Reproducibility notes

- **GPT-5.5**: The model is pinned to `gpt-5.5-2026-04-23`. Reasoning effort is set to the
  API default. GPT outputs are not bit-for-bit reproducible across runs, but should be
  stable given the pinned snapshot.
- **BERT training**: Random seed 0 controls the internal dev split; the 15-epoch ceiling
  with early stopping means the exact epoch count reached is determined by validation loss
  and is not recorded in the saved model files.
- **Train/test splits**: Stage 1 split is fully deterministic (greedy balancing, no seed).
  Stage 2 split uses seed=42 (`scripts/annotation/convert_annotations.py`).
- **Tesseract**: Version 5.5.2, language pack `nld+eng`, called via pytesseract.

---

## Software

| Package | Version | Purpose |
|---|---|---|
| Python | 3.13.13 | Runtime |
| PyTorch | 2.12.0 | Machine learning framework |
| Transformers | 5.10.1 | RobBERT model and fine-tuning |
| Tesseract | 5.5.2 | OCR for dossiers with missing text layers |
| Datasets | 4.8.5 | Data management for training |
| Accelerate | 1.13.0 | Device placement |
| ANLS_star | 1.0.1 | Calculation of ANLS* metric |
| Scikit-learn | 1.9.0 | Calculation of Cohen's Kappa score |
| PyMuPDF | 1.27.2 | PDF page rendering for OCR |
| Auxiliary | — | pdfplumber, pytesseract, Pillow, openai |
