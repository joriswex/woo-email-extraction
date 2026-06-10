"""
BERT Stage-1 Line-Level Training — Kaggle Notebook
====================================================
Trains RobBERT for email boundary detection using LINE-LEVEL binary
classification. Each line is classified as "email start" (1) or "not" (0),
with ±2 surrounding context lines included. This eliminates the 512-token
sliding window problem that caused over-segmentation in the token approach.

Produces the "BERT (line+filter)" result row in Stage 1 evaluation.

v2 (2026-06-09): retrained on expanded dataset — 10 train dossiers / 1 006 emails
(up from 6 dossiers in v1). Split is greedy email-count balanced across 20 dossiers.

Files to upload to your woolens-data Kaggle dataset:
  data/annotations/stage1_train.json   ← 10 dossiers, 1 006 emails
  src/bert_s1_data.py
  src/bert_tokenize.py
  src/bert_s1_train.py
  src/focal_loss.py

After training, download the zip and evaluate locally:
  huggingface-cli download joriswechs/woolens-stage2 stage1_line_v2_final.zip --local-dir models/
  unzip models/stage1_line_v2_final.zip -d models/stage1_line_v2
  python scripts/evaluate.py --stages 1 --bert-s1 models/stage1_line_v2 --bert-s1-label "BERT (line+filter)"
"""

# ── Cell 1: dependencies ───────────────────────────────────────────────────
# !pip install -q datasets accelerate

# ── Cell 2: setup — copy flat src/ files and add to path ──────────────────
import json, sys, shutil
from pathlib import Path

DATASET    = Path("/kaggle/input/datasets/joriswechsler/woolens-data")
DATA_PATH  = DATASET / "stage1_train.json"
OUTPUT_DIR = Path("/kaggle/working/stage1_line_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Copy src modules to a flat working directory and add to Python path
src_dir = Path("/kaggle/working/src")
src_dir.mkdir(exist_ok=True)
for fname in ["bert_s1_data.py", "bert_tokenize.py", "bert_s1_train.py", "focal_loss.py"]:
    f = DATASET / fname
    if f.exists():
        shutil.copy(f, src_dir / fname)
    else:
        print(f"WARNING: {fname} not found in dataset — upload it first")
sys.path.insert(0, str(src_dir))

# ── Cell 3: internal train / val split (2 dossiers held out for validation) ─
import random

with open(DATA_PATH) as f:
    all_records = json.load(f)

rng = random.Random(0)
shuffled = all_records[:]
rng.shuffle(shuffled)

val_records   = shuffled[:2]   # 2 dossiers for validation (20%) — more robust signal
train_records = shuffled[2:]   # 8 dossiers for training

(OUTPUT_DIR / "tmp_train.json").write_text(json.dumps(train_records, ensure_ascii=False))
(OUTPUT_DIR / "tmp_val.json").write_text(json.dumps(val_records,   ensure_ascii=False))

print(f"Training on {len(train_records)} dossiers, validating on {len(val_records)}")
print(f"Val dossiers: {[r['dossier_id'] for r in val_records]}")

# ── Cell 4: train ─────────────────────────────────────────────────────────
from bert_s1_train import train

train(
    train_path=OUTPUT_DIR / "tmp_train.json",
    dev_path=OUTPUT_DIR   / "tmp_val.json",
    output_dir=OUTPUT_DIR,
    num_epochs=15,               # ceiling; early stopping halts earlier if val loss plateaus
    batch_size=32,               # lines are short — larger batch is fine on T4
    early_stopping_patience=3,   # lower than stage 2 (binary task converges faster)
)

# ── Cell 5: cleanup ────────────────────────────────────────────────────────
(OUTPUT_DIR / "tmp_train.json").unlink(missing_ok=True)
(OUTPUT_DIR / "tmp_val.json").unlink(missing_ok=True)

# ── Cell 6: zip and upload to HuggingFace ─────────────────────────────────
import zipfile, os
from huggingface_hub import HfApi

with zipfile.ZipFile("/kaggle/working/stage1_line_v2_final.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for f in os.listdir(OUTPUT_DIR):
        if not f.startswith("checkpoint"):
            full = OUTPUT_DIR / f
            if full.is_file():
                zf.write(full, f)
                print(f"Added: {f}")

api = HfApi(token="YOUR_TOKEN_HERE")
api.upload_file(
    path_or_fileobj="/kaggle/working/stage1_line_v2_final.zip",
    path_in_repo="stage1_line_v2_final.zip",
    repo_id="joriswechs/woolens-stage2",
    repo_type="model",
)
print("Done — download with:")
print("  huggingface-cli download joriswechs/woolens-stage2 stage1_line_v2_final.zip --local-dir models/")
