# Label Studio Annotation Workflow

Two approaches are available. **Combined (recommended)** annotates boundaries
and field values in one pass per dossier. The original two-pass approach is kept
for reference.

---

## Prerequisites

```bash
pip install label-studio
label-studio start          # opens http://localhost:8080 in your browser
```

Create a free account when prompted.

---

## Recommended: Combined single-pass annotation

Annotate email boundaries AND field values (FROM, DATE, SUBJECT, …) in one go.
No handoff step needed between stages.

### 1. Create a project

1. Click **Create Project** → give it a name (e.g. "Woo Combined")
2. Go to **Labeling Setup** → switch to **Code** view → paste the contents of
   `labeling_config_combined.xml`
3. Save

### 2. Generate import tasks

```bash
python scripts/run_inference.py path/to/dossier.pdf \
    --output data/real/my_dossier_inference.json
```

Or use the existing import files already created under `data/real/`.

Import tasks are plain JSON with `text` and `dossier_id`:
```json
[{"data": {"text": "...", "dossier_id": "woo-1234"}}]
```

Go to **Import** → upload the file.

### 3. Annotate (for each email in the dossier)

Work through the dossier top to bottom. For each email:

1. Press `0`, drag to select the **full email** from the first header line to
   the last line of the body or signature → **EMAIL**
2. Press `f`, highlight the sender value after `Van:` or `From:` → **FROM**
   (if redacted, press `r` instead → **REDACTED**)
3. Press `t`, highlight the recipient value after `Aan:` or `To:` → **TO**
4. Press `c` for CC recipients if present → **CC**
5. Press `d`, highlight the date value after `Datum:` or `Sent:` → **DATE**
6. Press `s`, highlight the subject value after `Onderwerp:` or `Subject:` → **SUBJECT**
7. Press `b`, highlight the body text → **BODY**
8. Press `g`, highlight the closing greeting + name if present → **SIG**

**Important:** highlight only the VALUE, not the keyword.
- Correct: `Van: ` **`Jan de Vries <jan@bzk.nl>`**
- Wrong:   **`Van: Jan de Vries <jan@bzk.nl>`**

Email type (NEW / REPLY / FORWARD) is inferred automatically from the SUBJECT
value — no manual selection needed.

### 4. Export

Projects → Export → **JSON** → download. Save to `data/real/`.

### 5. Convert to internal formats (both stages at once)

```bash
python scripts/labelstudio/convert_labelstudio_export.py \
    --stage combined \
    --input  data/real/combined_export.json \
    --output data/annotations/stage1.json \
    --output_stage2 data/annotations/stage2.json
```

This produces:
- `stage1.json` — dossier-level email boundary annotations (for training the boundary model)
- `stage2.json` — email-level field annotations (for training the field-extraction model)

---

## Training after annotation

```bash
# Stage-1 model (email boundary detection)
python -c "
from woolens_email.train import train
train('data/annotations/stage1.json',
      'data/annotations/stage1_dev.json',
      'models/stage1/')
"

# Stage-2 model (field extraction)
python -c "
from woolens_email.train_field import train
train('data/annotations/stage2.json',
      'data/annotations/stage2_dev.json',
      'models/stage2/')
"
```

---

## Alternative: Original two-pass workflow

Use this if you prefer to annotate boundaries and fields separately, or if
different people are doing each annotation task.

### Pass 1 — Email boundary annotation

1. Create a project with `labeling_config_stage1.xml`
2. Import dossier text tasks
3. Annotate: highlight each email, label **EMAIL** (hotkey `e`)
4. Export → convert:
   ```bash
   python scripts/labelstudio/convert_labelstudio_export.py \
       --stage 1 \
       --input  stage1_export.json \
       --output data/annotations/stage1.json
   ```

### Handoff to pass 2

```bash
python scripts/labelstudio/export_stage1_to_stage2.py \
    --input      data/annotations/stage1.json \
    --output_dir data/labelstudio/stage2_import/
```

Writes one JSON file per dossier, one task per email.

### Pass 2 — Field-level annotation

1. Create a project with `labeling_config_stage2.xml`
2. Import files from `data/labelstudio/stage2_import/`
3. Annotate field spans + email type per email
4. Export → convert:
   ```bash
   python scripts/labelstudio/convert_labelstudio_export.py \
       --stage 2 \
       --input  stage2_export.json \
       --output data/annotations/stage2.json
   ```

---

## Full combined workflow summary

```
Real dossier PDFs
      │
      ▼ pdf_extract.extract_text()
Dossier text  →  Label Studio import task JSON
      │
      ▼ Label Studio (combined project, labeling_config_combined.xml)
      │  Annotate: EMAIL boundaries + FROM/TO/DATE/SUBJECT/BODY/SIG/REDACTED
      │
      ▼ convert_labelstudio_export.py --stage combined
      │
      ├── data/annotations/stage1.json  →  train.train()        → models/stage1/
      └── data/annotations/stage2.json  →  train_field.train()  → models/stage2/
```
