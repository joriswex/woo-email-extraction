"""
Line-level email boundary detection (Stage 1 alternative architecture).

Instead of token classification over full dossier text with a sliding window,
this approach classifies each line as "email start" (1) or "not" (0).
Each BERT call sees only the current line + a small context window of surrounding
lines, eliminating the 512-token sliding window problem that caused BERT's
over-segmentation.

Architecture: AutoModelForSequenceClassification (binary: 0=not start, 1=start)

Why this is better for Stage 1:
- No sliding window → no context loss mid-email
- Each prediction is localised: "does this line look like an email header?"
- BERT excels at binary sentence-level classification
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from focal_loss import FocalLoss

PRIMARY_MODEL = "DTAI-KULeuven/robbert-2023-dutch-base"
FALLBACK_MODEL = "pdelobelle/robbert-v2-dutch-base"

CONTEXT_LINES = 2   # lines of context before and after the target line
MAX_LENGTH    = 256  # lines are short; 256 tokens is ample
LABEL_START   = 1
LABEL_OTHER   = 0


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def _line_contexts(dossier_text: str, email_spans: list[tuple[int, int]]) -> list[dict]:
    """
    Split dossier text into lines and label each as email start or not.

    Returns a list of dicts with 'text' (the context window string) and 'label'.
    """
    lines = dossier_text.split("\n")

    # Build cumulative char offsets for each line
    offsets: list[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1  # +1 for the '\n'

    email_start_chars = {span[0] for span in email_spans}

    examples = []
    for i, (line, start) in enumerate(zip(lines, offsets)):
        if not line.strip():
            continue  # skip blank lines

        label = LABEL_START if start in email_start_chars else LABEL_OTHER

        # Context window: CONTEXT_LINES before + current + CONTEXT_LINES after
        ctx_start = max(0, i - CONTEXT_LINES)
        ctx_end   = min(len(lines), i + CONTEXT_LINES + 1)
        context   = "\n".join(lines[ctx_start:ctx_end])

        examples.append({"text": context, "label": label, "line_idx": i, "char_offset": start})
    return examples


def build_hf_dataset(annotations: list[dict], tokenizer) -> Dataset:
    """
    Convert dossier-level stage-1 annotations to a line-level HF Dataset.
    Columns: input_ids, attention_mask, label.
    """
    rows: list[dict] = []
    for ann in annotations:
        email_spans = [(e["start_char"], e["end_char"]) for e in ann["emails"]]
        for ex in _line_contexts(ann["text"], email_spans):
            encoded = tokenizer(
                ex["text"],
                max_length=MAX_LENGTH,
                truncation=True,
                padding=False,
            )
            rows.append({
                "input_ids":      encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "label":          ex["label"],
            })

    # Log class distribution
    n_start = sum(1 for r in rows if r["label"] == LABEL_START)
    print(f"  Line dataset: {len(rows)} lines, {n_start} email starts ({n_start/len(rows)*100:.1f}%)")
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Model + trainer
# ---------------------------------------------------------------------------


class _FocalTrainer(Trainer):
    """Trainer with Focal Loss and class weights for imbalanced binary classification."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        weight  = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fn = FocalLoss(gamma=2.0, weight=weight, ignore_index=-100)
        loss    = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def _load_model_and_tokenizer(model_name: str = PRIMARY_MODEL):
    for name in [model_name, FALLBACK_MODEL]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(name)
            model     = AutoModelForSequenceClassification.from_pretrained(
                name, num_labels=2, ignore_mismatched_sizes=True
            )
            if name != model_name:
                print(f"WARNING: using fallback '{name}'")
            return model, tokenizer
        except Exception as exc:
            if name == FALLBACK_MODEL:
                raise RuntimeError(f"Both models failed to load.") from exc
            print(f"WARNING: '{name}' failed: {exc}")


def _compute_class_weights(dataset: Dataset) -> torch.Tensor:
    counts = [0, 0]
    for ex in dataset:
        counts[ex["label"]] += 1
    total = sum(counts)
    weights = [(total / (2 * c)) ** 0.5 if c > 0 else 1.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train(
    train_path: str | Path,
    dev_path: str | Path,
    output_dir: str | Path,
    model_name: str = PRIMARY_MODEL,
    num_epochs: int = 5,
    batch_size: int = 16,
) -> None:
    """
    Fine-tune RobBERT for line-level email boundary detection.

    train_path / dev_path : stage-1 annotation JSON (array of dossier dicts)
    output_dir            : where to save the trained model
    """
    print("Loading model …")
    model, tokenizer = _load_model_and_tokenizer(model_name)

    print("Building line-level datasets …")
    with open(train_path) as f:
        train_ann = json.load(f)
    with open(dev_path) as f:
        dev_ann = json.load(f)

    train_ds = build_hf_dataset(train_ann, tokenizer)
    dev_ds   = build_hf_dataset(dev_ann,   tokenizer)
    print(f"  train lines: {len(train_ds)}  dev lines: {len(dev_ds)}")

    class_weights = _compute_class_weights(train_ds)
    print(f"  class weights (not-start, start): {class_weights.tolist()}")

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50,
        report_to="none",
    )

    trainer = _FocalTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        class_weights=class_weights,
    )

    print("Training …")
    trainer.train()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved to {output_dir}")
