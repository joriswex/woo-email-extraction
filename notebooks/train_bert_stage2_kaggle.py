"""
BERT Stage-2 Field Extraction Training — Kaggle Notebook
=========================================================
Trains RobBERT for field extraction using Focal Loss (gamma=2.0).

Focal Loss focuses training on hard, uncertain tokens and naturally
down-weights easy O tokens that dominate the sequence — no manual
class weights needed.

Files to upload to woolens-data Kaggle dataset:
  data/annotations/stage2_train.json
  src/bert_s2_data.py
  src/bert_tokenize.py
  src/bert_s2_train.py
  src/focal_loss.py
"""

# ── Cell 1: dependencies ───────────────────────────────────────────────────
# !pip install -q datasets accelerate

# ── Cell 2: setup ─────────────────────────────────────────────────────────
import json, sys, shutil
from pathlib import Path

DATASET    = Path("/kaggle/input/datasets/joriswechsler/woolens-data")
DATA_PATH  = DATASET / "stage2_train.json"
OUTPUT_DIR = Path("/kaggle/working/stage2_focal_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

src_dir = Path("/kaggle/working/src")
src_dir.mkdir(exist_ok=True)
for fname in ["bert_s2_data.py", "bert_tokenize.py", "bert_s2_train.py", "focal_loss.py"]:
    src = DATASET / fname
    if src.exists():
        shutil.copy(src, src_dir / fname)
    else:
        print(f"WARNING: {fname} not found in dataset")
sys.path.insert(0, "/kaggle/working/src")

# ── Cell 3: internal train / val split ────────────────────────────────────
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

# ── Cell 4: train ─────────────────────────────────────────────────────────
from bert_s2_train import train

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
print("hf download joriswechs/woolens-stage2 stage2_focal_v1_final.zip --local-dir models/")
