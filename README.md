# Information Extraction from Dutch Woo Dossiers

This repository contains the code and data for a thesis project comparing three approaches to information extraction from Dutch government Woo dossiers: a regex baseline, a fine-tuned RobBERT model, and GPT-5.5. The task is split into two stages — email boundary detection (Stage 1) and field extraction (Stage 2) — evaluated on a manually annotated corpus of emails from Woo PDF dossiers.

## Repository structure

```
Thesis_Project_Clean/
├── src/                    # Flat module directory (add to sys.path)
│   ├── regex_baseline.py   # Regex pipeline for boundary detection and field extraction
│   ├── gpt_baseline.py     # GPT-5.5 pipeline (vision + text)
│   ├── bert_s1_train.py    # Stage-1 BERT training (line-level classifier)
│   ├── bert_s1_predict.py  # Stage-1 BERT inference + proximity filter
│   ├── bert_s2_train.py    # Stage-2 BERT training (token classification)
│   ├── bert_s2_predict.py  # Stage-2 BERT inference
│   ├── bert_s1_data.py     # Stage-1 annotation schema and BIO label conversion
│   ├── bert_s2_data.py     # Stage-2 field label schema and BIO label conversion
│   ├── bert_tokenize.py    # Sliding-window tokenisation for long documents
│   ├── focal_loss.py       # Focal Loss for imbalanced token classification
│   ├── eval_metrics.py     # Shared evaluation metrics (span IoU, exact match, ANLS)
│   └── pdf_extract.py      # PDF text extraction via pdfplumber
├── scripts/
│   └── evaluate.py         # Main evaluation script (all three approaches)
├── notebooks/
│   ├── train_bert_stage1_kaggle.py   # Kaggle notebook for Stage-1 BERT training
│   └── train_bert_stage2_kaggle.py   # Kaggle notebook for Stage-2 BERT training
├── data/
│   ├── annotations/        # JSON annotation files (train/test splits)
│   └── results/            # Evaluation output CSVs
├── .gitignore
├── README.md
└── results.md
```

## Three pipelines

**Regex** — Pattern-based extraction using Dutch/English email header keywords (Van:, From:, Aan:, To:, etc.). Multi-line recipient lists and attachment filename detection without a Bijlagen: keyword are supported. Unique fields use first-occurrence deduplication to avoid picking up forwarded-chain headers.

**BERT (RobBERT)** — Fine-tuned `DTAI-KULeuven/robbert-2023-dutch-base` model. Stage 1 uses line-level binary classification (email start vs. not) with a proximity filter to collapse multiple predictions within the same header block. Stage 2 uses token-level BIO classification with Focal Loss to handle class imbalance across field types.

**GPT-5.5 (vision)** — Calls the GPT-5.5-2026-04-23 API with both PDF page images and extracted text. Stage 1 uses text-only boundary detection; Stage 2 uses vision-augmented field extraction (prompt version 3) that handles heavy redaction and OCR artefacts common in Woo documents.

## How to run evaluation

```bash
# Regex only (default — no API key needed)
python scripts/evaluate.py

# Add GPT-5.5 vision (requires OPENAI_API_KEY)
python scripts/evaluate.py --gpt

# Full comparison with BERT checkpoints
python scripts/evaluate.py --gpt --bert-s1 models/stage1 --bert-s2 models/stage2

# Stage 2 only with ANLS computation
python scripts/evaluate.py --stages 2 --anls

# All flags
python scripts/evaluate.py [--gpt] [--bert-s1 PATH] [--bert-s2 PATH] [--stages {1,2}] [--anls] [--output-dir PATH]
```

Results are written to `data/results/`. The `--anls` flag computes official ANLS* scores using the `anls_star` package after evaluation and writes `data/results/stage2_anls.csv`.

## How to train BERT

Training notebooks are in `notebooks/` and are designed to run on Kaggle (GPU required):

- `train_bert_stage1_kaggle.py` — trains the Stage-1 line-level classifier on `stage1_train.json`
- `train_bert_stage2_kaggle.py` — trains the Stage-2 field extractor on `stage2_train.json`

Upload the relevant `src/` files and annotation JSON to a Kaggle dataset (`woolens-data`) before running. Both notebooks include a cell to zip and upload the trained model to HuggingFace.

## Data

Annotations are stored as JSON arrays in `data/annotations/`:

- `stage1_train.json` / `stage1_test.json` — dossier-level annotations with email boundary spans `[{"email_id": ..., "start_char": ..., "end_char": ...}]`
- `stage2_train.json` / `stage2_test.json` — email-level annotations with field spans `[{"label": "FROM"|"TO"|"CC"|"DATE"|"SUBJECT"|"ATTACHMENT", "start_char": ..., "end_char": ...}]`

All character offsets are relative to the extracted dossier text (Stage 1) or individual email text (Stage 2).

## Requirements

```
transformers
torch
openai
pdfplumber
datasets
anls_star
```

Install with:
```bash
pip install transformers torch openai pdfplumber datasets anls_star
```
