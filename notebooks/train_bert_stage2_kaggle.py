"""
BERT Stage-2 Field Extraction — Unweighted & Weighted Focal Loss (Kaggle)
=========================================================================
Trains RobBERT twice in sequence using identical hyperparameters, producing
two models for direct comparison in the thesis evaluation table:

  - stage2_focal_v2   : plain Focal Loss, no class balancing (baseline)
  - stage2_weighted_v2: Focal Loss with sqrt-inverse-frequency class weights

Both models use the same architecture, data, epochs, and early-stopping rule
so any difference in results is attributable solely to the class-weighting
strategy.  The weighted model is expected to recover from the F1=0.000 on
CC, SUBJECT, and ATTACHMENT that the unweighted model exhibits due to
label collapse from extreme class imbalance (~70 % O tokens).

v2 (2026-06-09): trained on expanded dataset — 1 006 train records across
20 dossiers (up from 683 records across 8 dossiers in v1).

Files to upload to your woolens-data Kaggle dataset:
  data/annotations/stage2_train.json   ← 1 006 records, 20 dossiers
  src/bert_s1_data.py
  src/bert_s2_data.py
  src/bert_tokenize.py
  src/bert_s2_train.py
  src/focal_loss.py

After training, download both zips and evaluate locally:
  huggingface-cli download joriswechs/woolens-stage2 stage2_focal_v2_final.zip --local-dir models/
  huggingface-cli download joriswechs/woolens-stage2 stage2_weighted_v2_final.zip --local-dir models/
  unzip models/stage2_focal_v2_final.zip -d models/stage2_focal_v2
  unzip models/stage2_weighted_v2_final.zip -d models/stage2_weighted_v2
  python scripts/evaluate.py --stages 2 \\
      --bert-s2 models/stage2_focal_v2    --bert-s2-label "BERT (unweighted)" \\
      --bert-s2 models/stage2_weighted_v2 --bert-s2-label "BERT (weighted)"
"""

# ── Cell 1: dependencies ───────────────────────────────────────────────────
# !pip install -q datasets accelerate

# ── Cell 2: setup — copy flat src/ files and add to path ──────────────────
import json, sys, shutil
from pathlib import Path

DATASET   = Path("/kaggle/input/datasets/joriswechsler/woolens-data")
DATA_PATH = DATASET / "stage2_train.json"

src_dir = Path("/kaggle/working/src")
src_dir.mkdir(exist_ok=True)
for fname in ["bert_s1_data.py", "bert_s2_data.py", "bert_tokenize.py", "bert_s2_train.py", "focal_loss.py"]:
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

print(f"Training on {len(train_records)} emails, validating on {len(val_records)}")

# Write shared split files — both models train on exactly the same data
Path("/kaggle/working/tmp_train.json").write_text(json.dumps(train_records, ensure_ascii=False))
Path("/kaggle/working/tmp_val.json").write_text(json.dumps(val_records, ensure_ascii=False))

# ── Cell 4: train unweighted model (plain Focal Loss — baseline) ───────────
from bert_s2_train import train
import os, zipfile
from huggingface_hub import HfApi

FOCAL_DIR = Path("/kaggle/working/stage2_focal_v2")
for _d in FOCAL_DIR.glob("checkpoint-*") if FOCAL_DIR.exists() else []:
    shutil.rmtree(_d)
FOCAL_DIR.mkdir(parents=True, exist_ok=True)

print("\n=== Training: UNWEIGHTED (plain Focal Loss) ===")
train(
    train_path=Path("/kaggle/working/tmp_train.json"),
    dev_path=Path("/kaggle/working/tmp_val.json"),
    output_dir=FOCAL_DIR,
    num_epochs=15,
    batch_size=16,
    early_stopping_patience=5,
    use_class_weights=False,
)

# ── Cell 5: zip + upload unweighted, then free disk space ─────────────────
focal_zip = "/kaggle/working/stage2_focal_v2_final.zip"
with zipfile.ZipFile(focal_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in os.listdir(FOCAL_DIR):
        if not f.startswith("checkpoint"):
            full = FOCAL_DIR / f
            if full.is_file():
                zf.write(full, f)
                print(f"Added: {f}")

api = HfApi(token="YOUR_TOKEN_HERE")
api.upload_file(
    path_or_fileobj=focal_zip,
    path_in_repo="stage2_focal_v2_final.zip",
    repo_id="joriswechs/woolens-stage2",
    repo_type="model",
)
print("Uploaded stage2_focal_v2")

# Free disk space and GPU memory before training the second model
shutil.rmtree(FOCAL_DIR)
os.remove(focal_zip)

import gc, torch
gc.collect()
torch.cuda.empty_cache()
print("GPU memory cleared for second training run")

# ── Cell 6: train weighted model (class-weighted Focal Loss) ───────────────
WEIGHTED_DIR = Path("/kaggle/working/stage2_weighted_v2")
for _d in WEIGHTED_DIR.glob("checkpoint-*") if WEIGHTED_DIR.exists() else []:
    shutil.rmtree(_d)
WEIGHTED_DIR.mkdir(parents=True, exist_ok=True)

print("\n=== Training: WEIGHTED (sqrt-inverse-frequency class weights) ===")
train(
    train_path=Path("/kaggle/working/tmp_train.json"),
    dev_path=Path("/kaggle/working/tmp_val.json"),
    output_dir=WEIGHTED_DIR,
    num_epochs=15,
    batch_size=16,
    early_stopping_patience=5,
    use_class_weights=True,
)

# ── Cell 7: zip + upload weighted ─────────────────────────────────────────
weighted_zip = "/kaggle/working/stage2_weighted_v2_final.zip"
with zipfile.ZipFile(weighted_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in os.listdir(WEIGHTED_DIR):
        if not f.startswith("checkpoint"):
            full = WEIGHTED_DIR / f
            if full.is_file():
                zf.write(full, f)
                print(f"Added: {f}")

api.upload_file(
    path_or_fileobj=weighted_zip,
    path_in_repo="stage2_weighted_v2_final.zip",
    repo_id="joriswechs/woolens-stage2",
    repo_type="model",
)
print("Uploaded stage2_weighted_v2")

# ── Cell 8: cleanup ────────────────────────────────────────────────────────
Path("/kaggle/working/tmp_train.json").unlink(missing_ok=True)
Path("/kaggle/working/tmp_val.json").unlink(missing_ok=True)

print("\nDone. Download with:")
print("  huggingface-cli download joriswechs/woolens-stage2 stage2_focal_v2_final.zip --local-dir models/")
print("  huggingface-cli download joriswechs/woolens-stage2 stage2_weighted_v2_final.zip --local-dir models/")
