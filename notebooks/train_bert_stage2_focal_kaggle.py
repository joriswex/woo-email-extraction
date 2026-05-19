"""
BERT Stage-2 Field Extraction — Focal Loss only, no class weights (Kaggle)
===========================================================================
Trains RobBERT for field extraction using Focal Loss (gamma=2.0) without
class weighting. This is the unweighted baseline that produces F1=0.000 on
CC, SUBJECT, and ATTACHMENT due to label collapse from class imbalance.

Produces the "BERT" (unweighted) result row in Stage 2 evaluation.
For the improved weighted version, use train_bert_stage2_weighted_kaggle.py.

Files to upload to your woolens-data Kaggle dataset (from src/):
  data/annotations/stage2_train.json
  src/bert_s2_data.py
  src/bert_tokenize.py
  src/bert_s2_train.py
  src/focal_loss.py

After training, evaluate locally:
  python scripts/evaluate.py --stages 2 --bert-s2 models/stage2_focal_v1
"""

# ── Cell 1: dependencies ───────────────────────────────────────────────────
# !pip install -q datasets accelerate

# ── Cell 2: setup — copy flat src/ files and add to path ──────────────────
import json, sys, shutil
from pathlib import Path

DATASET    = Path("/kaggle/input/datasets/joriswechsler/woolens-data")
DATA_PATH  = DATASET / "stage2_train.json"
OUTPUT_DIR = Path("/kaggle/working/stage2_focal_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Copy src modules to a flat working directory and add to Python path
src_dir = Path("/kaggle/working/src")
src_dir.mkdir(exist_ok=True)
for fname in ["bert_s2_data.py", "bert_tokenize.py", "bert_s2_train.py", "focal_loss.py"]:
    f = DATASET / fname
    if f.exists():
        shutil.copy(f, src_dir / fname)
    else:
        print(f"WARNING: {fname} not found in dataset — upload it first")
sys.path.insert(0, str(src_dir))

# ── Cell 3: internal train / val split (10% held out for validation) ───────
import random

with open(DATA_PATH) as f:
    all_records = json.load(f)

rng = random.Random(0)
shuffled = all_records[:]
rng.shuffle(shuffled)

n_val         = max(1, round(len(shuffled) * 0.10))
val_records   = shuffled[:n_val]
train_records = shuffled[n_val:]

(OUTPUT_DIR / "tmp_train.json").write_text(json.dumps(train_records, ensure_ascii=False))
(OUTPUT_DIR / "tmp_val.json").write_text(json.dumps(val_records,   ensure_ascii=False))

print(f"Training on {len(train_records)} emails, validating on {len(val_records)}")

# ── Cell 4: train (no class weights — unweighted baseline) ────────────────
from bert_s2_train import train

# Note: overrides default num_epochs=15 and early_stopping to match the
# original unweighted training run (5 epochs, no early stopping)
train(
    train_path=OUTPUT_DIR / "tmp_train.json",
    dev_path=OUTPUT_DIR   / "tmp_val.json",
    output_dir=OUTPUT_DIR,
    num_epochs=5,
    batch_size=16,
)

# ── Cell 5: cleanup ────────────────────────────────────────────────────────
(OUTPUT_DIR / "tmp_train.json").unlink(missing_ok=True)
(OUTPUT_DIR / "tmp_val.json").unlink(missing_ok=True)

# ── Cell 6: zip and upload to HuggingFace ─────────────────────────────────
import zipfile, os
from huggingface_hub import HfApi

with zipfile.ZipFile("/kaggle/working/stage2_focal_v1_final.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for f in os.listdir(OUTPUT_DIR):
        if not f.startswith("checkpoint"):
            full = OUTPUT_DIR / f
            if full.is_file():
                zf.write(full, f)
                print(f"Added: {f}")

api = HfApi(token="YOUR_TOKEN_HERE")
api.upload_file(
    path_or_fileobj="/kaggle/working/stage2_focal_v1_final.zip",
    path_in_repo="stage2_focal_v1_final.zip",
    repo_id="joriswechs/woolens-stage2",
    repo_type="model",
)
print("Done — download with:")
print("  huggingface-cli download joriswechs/woolens-stage2 stage2_focal_v1_final.zip --local-dir models/")
