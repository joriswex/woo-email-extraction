# Information Extraction from Dutch Woo Dossiers

This repository contains the code and data for a master's thesis comparing three approaches to information extraction from Dutch government Woo (*Wet open overheid*) dossier PDFs: a regex baseline, a fine-tuned RobBERT model, and GPT-5.5. The task is split into two stages — email boundary detection (Stage 1) and field extraction (Stage 2) — evaluated on a manually annotated corpus of 854 emails from 8 Woo PDF dossiers.

## Repository structure

```
Thesis_Project_Clean/
├── src/                              # Flat module directory (add to sys.path)
│   ├── regex_baseline.py             # Regex pipeline for boundary detection and field extraction
│   ├── gpt_baseline.py               # GPT-5.5 pipeline (text + vision modes for both stages)
│   ├── bert_s1_train.py              # Stage-1 BERT training (line-level classifier)
│   ├── bert_s1_predict.py            # Stage-1 BERT inference + proximity filter
│   ├── bert_s2_train.py              # Stage-2 BERT training (token classification + class weights)
│   ├── bert_s2_predict.py            # Stage-2 BERT inference
│   ├── bert_s1_data.py               # Stage-1 annotation schema and BIO label conversion
│   ├── bert_s2_data.py               # Stage-2 field label schema and BIO label conversion
│   ├── bert_tokenize.py              # Sliding-window tokenisation for long documents
│   ├── focal_loss.py                 # Focal Loss with class weights for imbalanced classification
│   ├── eval_metrics.py               # Shared evaluation metrics (span IoU, exact match, ANLS)
│   └── pdf_extract.py                # PDF text extraction via pdfplumber
├── scripts/
│   └── evaluate.py                   # Main evaluation script (all approaches, both stages)
├── notebooks/
│   ├── train_bert_stage1_kaggle.py         # Stage-1 token BIO classifier (Kaggle)
│   ├── train_bert_stage1_line_kaggle.py    # Stage-1 line-level classifier (Kaggle)
│   ├── train_bert_stage2_kaggle.py         # Stage-2 field extractor, basic (Kaggle)
│   ├── train_bert_stage2_focal_kaggle.py   # Stage-2 with Focal Loss (Kaggle)
│   └── train_bert_stage2_weighted_kaggle.py # Stage-2 with class-weighted Focal Loss (Kaggle)
├── data/
│   ├── annotations/                  # JSON annotation files (train/test splits)
│   └── results/                      # Evaluation output CSVs and raw predictions
├── .gitignore
├── README.md
└── results.md
```

## Three pipelines

**Regex** — Pattern-based extraction using Dutch/English email header keywords (`Van:`, `From:`, `Aan:`, `To:`, etc.). Multi-line recipient lists and attachment filename detection are supported. Unique fields use first-occurrence deduplication to avoid picking up forwarded-chain headers.

**BERT (RobBERT)** — Fine-tuned `DTAI-KULeuven/robbert-2023-dutch-base`. Stage 1 uses line-level binary classification (email start vs. not) with a proximity filter. Stage 2 uses BIO token classification with class-weighted Focal Loss (Lin et al., 2017; Cui et al., 2019) to handle the severe token-level class imbalance inherent to email field labelling. Square-root inverse frequency weights are computed from the training data and passed as the α parameter of Focal Loss.

**GPT-5.5** — Calls the `gpt-5.5-2026-04-23` API (pinned for reproducibility). Both text-only and vision+text modes are evaluated for each stage:
- Stage 1 text: full dossier text, date-line anchoring strategy
- Stage 1 vision+text: all PDF pages at 72 DPI + dossier text; GPT identifies boundaries visually
- Stage 2 vision+text: email PDF pages at 150 DPI + email text; prompt v3 handles redaction and OCR artefacts
- Stage 2 text: email text only; direct ablation of the visual modality

## How to run evaluation

```bash
# Regex only (default — no API key needed)
python scripts/evaluate.py

# GPT-5.5 Stage 2 vision+text (requires OPENAI_API_KEY)
python scripts/evaluate.py --gpt --gpt-versions 3

# GPT-5.5 Stage 1 vision+text
python scripts/evaluate.py --stages 1 --gpt-s1-vision

# GPT-5.5 Stage 2 text-only
python scripts/evaluate.py --stages 2 --gpt-s2-text

# BERT with a custom label
python scripts/evaluate.py --stages 2 --bert-s2 models/stage2_weighted --bert-s2-label "BERT (weighted)"

# Stage 1 BERT line classifier
python scripts/evaluate.py --stages 1 --bert-s1 models/stage1_line --bert-s1-arch line

# All flags
python scripts/evaluate.py \
  [--stages {1,2}] \
  [--gpt] [--gpt-versions {1,2,3}] \
  [--gpt-s1-vision] [--gpt-s2-text] \
  [--bert-s1 PATH] [--bert-s1-arch {token,line}] \
  [--bert-s2 PATH] [--bert-s2-arch {token,crf}] [--bert-s2-label LABEL] \
  [--output-dir PATH]
```

Results are written to `data/results/` and automatically archived with a timestamp.

## How to train BERT

Training notebooks are in `notebooks/` and designed to run on Kaggle (GPU required). All notebooks upload the trained model to HuggingFace for download.

| Notebook | Description |
|---|---|
| `train_bert_stage1_line_kaggle.py` | Stage-1 line-level classifier with class-weighted Focal Loss |
| `train_bert_stage2_weighted_kaggle.py` | Stage-2 field extractor with sqrt inverse-frequency class weights + early stopping (**recommended**) |
| `train_bert_stage2_focal_kaggle.py` | Stage-2 with Focal Loss only (no class weights — baseline) |

Upload the relevant `src/` files and annotation JSON to a Kaggle dataset (`woolens-data`) before running.

## Data

Annotations are stored as JSON arrays in `data/annotations/`:

- `stage1_train.json` / `stage1_test.json` — dossier-level annotations with email boundary spans
- `stage2_train.json` / `stage2_test.json` — email-level annotations with field spans for FROM, TO, CC, DATE, SUBJECT, ATTACHMENT

All character offsets are relative to the pdfplumber-extracted text (Stage 1: dossier text; Stage 2: individual email text).

## Requirements

```
transformers>=4.40
torch>=2.0
openai
pdfplumber
datasets
accelerate
```

Install with:
```bash
pip install transformers torch openai pdfplumber datasets accelerate
```
