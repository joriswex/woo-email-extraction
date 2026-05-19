"""
BERT Stage-2 Field Extraction — Class-Weighted Focal Loss (Kaggle)
==================================================================
Trains RobBERT with sqrt-inverse-frequency class weights passed as the
alpha parameter of Focal Loss (Lin et al., 2017; Cui et al., 2019).

Why: the vanilla BERT run produced 0.000 F1 on CC, SUBJECT, and ATTACHMENT
because the unweighted loss was dominated by O tokens (70% of all tokens).
The sqrt class weights give B-ATTACHMENT ~39x, B-CC ~19x, B-SUBJECT ~13x
relative to O, breaking the label-collapse without the instability of raw
inverse-frequency weights.

Files to upload to your woolens-data Kaggle dataset:
  data/annotations/stage2_train.json
  src/woolens_email/__init__.py
  src/woolens_email/data_field.py
  src/woolens_email/tokenize.py
  src/woolens_email/train_field.py      ← updated (class weights wired in)
  src/woolens_email/focal_loss.py

After training, evaluate locally with:
  python scripts/evaluate.py --stages 2 --bert-s2 models/stage2_weighted_v1 --bert-s2-label "BERT (weighted)"
"""

# ── Cell 1: dependencies ───────────────────────────────────────────────────
# !pip install -q datasets accelerate

# ── Cell 2: setup ─────────────────────────────────────────────────────────
import json, sys, shutil
from pathlib import Path

DATASET    = Path("/kaggle/input/datasets/joriswechsler/woolens-data")
DATA_PATH  = DATASET / "stage2_train.json"
OUTPUT_DIR = Path("/kaggle/working/stage2_weighted_v1")
# Remove leftover checkpoints from previous runs to avoid disk-full errors.
if OUTPUT_DIR.exists():
    import shutil as _shutil
    for _d in OUTPUT_DIR.glob("checkpoint-*"):
        _shutil.rmtree(_d)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pkg = Path("/kaggle/working/woolens_email")
pkg.mkdir(exist_ok=True)
for fname in ["__init__.py", "data.py", "data_field.py", "tokenize.py",
              "train_field.py", "focal_loss.py"]:
    src = DATASET / fname
    if src.exists():
        shutil.copy(src, pkg / fname)
    else:
        print(f"WARNING: {fname} not found in dataset — upload it first")
sys.path.insert(0, "/kaggle/working")

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
from woolens_email.train_field import train

train(
    train_path=OUTPUT_DIR / "tmp_train.json",
    dev_path=OUTPUT_DIR   / "tmp_val.json",
    output_dir=OUTPUT_DIR,
    num_epochs=15,              # ceiling — early stopping will halt earlier if val loss plateaus
    batch_size=16,              # T4 has 15 GB VRAM; batch 16 keeps gradient estimates stable
    early_stopping_patience=3,  # stop if no val-loss improvement for 3 consecutive epochs
)

# ── Cell 5: cleanup ────────────────────────────────────────────────────────
(OUTPUT_DIR / "tmp_train.json").unlink(missing_ok=True)
(OUTPUT_DIR / "tmp_val.json").unlink(missing_ok=True)

# ── Cell 6: zip and upload to HuggingFace ─────────────────────────────────
import zipfile, os
from huggingface_hub import HfApi

zip_path = "/kaggle/working/stage2_weighted_v1_final.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in os.listdir(OUTPUT_DIR):
        if not f.startswith("checkpoint"):
            full = OUTPUT_DIR / f
            if full.is_file():
                zf.write(full, f)
                print(f"Added: {f}")

api = HfApi(token="YOUR_TOKEN_HERE")
api.upload_file(
    path_or_fileobj=zip_path,
    path_in_repo="stage2_weighted_v1_final.zip",
    repo_id="joriswechs/woolens-stage2",
    repo_type="model",
)
print("Done — download with:")
print("  huggingface-cli download joriswechs/woolens-stage2 stage2_weighted_v1_final.zip --local-dir models/")
