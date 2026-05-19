"""
BERT Stage-1 Line-Level Training — Kaggle Notebook
====================================================
Trains RobBERT for email boundary detection using LINE-LEVEL classification
instead of token classification.

Each line is classified as "email start" (1) or "not" (0), with surrounding
context lines included. This eliminates the 512-token sliding window problem
that caused over-segmentation in the token-classification approach.

Setup: same as other notebooks — add train_line.py and focal_loss.py to the
woolens-data Kaggle dataset alongside stage1_train.json.

Files to upload to woolens-data dataset:
  data/annotations/stage1_train.json
  src/woolens_email/__init__.py
  src/woolens_email/data.py
  src/woolens_email/tokenize.py
  src/woolens_email/train_line.py
  src/woolens_email/focal_loss.py
"""

# ── Cell 1: dependencies ───────────────────────────────────────────────────
# !pip install -q datasets accelerate

# ── Cell 2: setup ─────────────────────────────────────────────────────────
import json, sys, shutil
from pathlib import Path

DATASET    = Path("/kaggle/input/datasets/joriswechsler/woolens-data")
DATA_PATH  = DATASET / "stage1_train.json"
OUTPUT_DIR = Path("/kaggle/working/stage1_line_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pkg = Path("/kaggle/working/woolens_email")
pkg.mkdir(exist_ok=True)
for fname in ["__init__.py", "data.py", "tokenize.py", "train_line.py", "focal_loss.py"]:
    src = DATASET / fname
    if src.exists():
        shutil.copy(src, pkg / fname)
sys.path.insert(0, "/kaggle/working")

# ── Cell 3: internal train / val split (1 dossier for val) ────────────────
import random

with open(DATA_PATH) as f:
    all_records = json.load(f)

rng = random.Random(0)
shuffled = all_records[:]
rng.shuffle(shuffled)

val_records   = shuffled[:1]
train_records = shuffled[1:]

(OUTPUT_DIR / "tmp_train.json").write_text(json.dumps(train_records, ensure_ascii=False))
(OUTPUT_DIR / "tmp_val.json").write_text(json.dumps(val_records,   ensure_ascii=False))

print(f"Training on {len(train_records)} dossiers, validating on {len(val_records)}")
print(f"Val dossier: {[r['dossier_id'] for r in val_records]}")

# ── Cell 4: train ─────────────────────────────────────────────────────────
from woolens_email.train_line import train

train(
    train_path=OUTPUT_DIR / "tmp_train.json",
    dev_path=OUTPUT_DIR   / "tmp_val.json",
    output_dir=OUTPUT_DIR,
    num_epochs=5,
    batch_size=32,  # lines are short — can use larger batch
)

# ── Cell 5: cleanup ────────────────────────────────────────────────────────
(OUTPUT_DIR / "tmp_train.json").unlink(missing_ok=True)
(OUTPUT_DIR / "tmp_val.json").unlink(missing_ok=True)

# ── Cell 6: zip and upload to HuggingFace ─────────────────────────────────
import zipfile, os
from huggingface_hub import HfApi

with zipfile.ZipFile("/kaggle/working/stage1_line_v1_final.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for f in os.listdir(OUTPUT_DIR):
        if not f.startswith("checkpoint"):
            full = OUTPUT_DIR / f
            if full.is_file():
                zf.write(full, f)
                print(f"Added: {f}")

api = HfApi(token="YOUR_TOKEN_HERE")
api.upload_file(
    path_or_fileobj="/kaggle/working/stage1_line_v1_final.zip",
    path_in_repo="stage1_line_v1_final.zip",
    repo_id="joriswechs/woolens-stage2",
    repo_type="model",
)
print("Done — download with:")
print("hf download joriswechs/woolens-stage2 stage1_line_v1_final.zip --local-dir models/")
