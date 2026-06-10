"""
HuggingFace Trainer setup for email field extraction (stage 2).

Model: AutoModelForTokenClassification (19 labels: B-/I- for each field type + O)
Base:  DTAI-KULeuven/robbert-2023-dutch-base (falls back to pdelobelle/robbert-v2-dutch-base)

Input: email-level annotation JSON (stage-2 schema), one email per record.
Each email is tokenised with stride=64 (smaller than the dossier-level stride=128
because emails are short and field precision matters more than window throughput).
"""

from __future__ import annotations
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from bert_s2_data import LABEL2ID, ID2LABEL, IGNORED, annotate_window
from focal_loss import FocalLoss
from bert_tokenize import tokenize_document

PRIMARY_MODEL = "DTAI-KULeuven/robbert-2023-dutch-base"
FALLBACK_MODEL = "pdelobelle/robbert-v2-dutch-base"

_FIELD_STRIDE = 64


def _compute_class_weights(dataset: Dataset) -> torch.Tensor:
    """Sqrt-inverse-frequency class weights (Mikolov et al., 2013 / Cui et al., 2019)."""
    n_labels = len(LABEL2ID)
    counts = [0] * n_labels
    for ex in dataset:
        for label in ex["labels"]:
            if label >= 0:
                counts[label] += 1
    total = sum(counts)
    weights = [(total / (n_labels * c)) ** 0.5 if c > 0 else 1.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float)


def load_model_and_tokenizer(model_name: str = PRIMARY_MODEL) -> tuple:
    """
    Load tokenizer and token-classification model for field extraction.
    Falls back gracefully. Returns (model, tokenizer).
    """
    for name in [model_name, FALLBACK_MODEL]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForTokenClassification.from_pretrained(
                name,
                num_labels=len(LABEL2ID),
                id2label=ID2LABEL,
                label2id=LABEL2ID,
                ignore_mismatched_sizes=True,
            )
            if name != model_name:
                print(f"WARNING: '{model_name}' unavailable; using fallback '{name}'")
            return model, tokenizer
        except Exception as exc:
            if name == model_name:
                print(f"WARNING: could not load '{name}': {exc}")
                print(f"  → attempting fallback '{FALLBACK_MODEL}'")
            else:
                raise RuntimeError(
                    f"Both '{model_name}' and '{FALLBACK_MODEL}' failed to load."
                ) from exc


def load_annotations(path: str | Path) -> list[dict]:
    """Load a JSON array of stage-2 email annotation dicts from disk."""
    with open(path) as f:
        return json.load(f)


class _FocalLossTrainer(Trainer):
    """Trainer subclass that applies class-weighted Focal Loss for handling class imbalance."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = FocalLoss(gamma=2.0, weight=self.class_weights, ignore_index=IGNORED)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def build_hf_dataset(annotations: list[dict], tokenizer) -> Dataset:
    """
    Convert stage-2 email annotation dicts into a HF Dataset of tokenizer windows.

    Uses stride=64 (smaller than dossier-level stride=128) because emails are
    short and precise field boundaries matter more than window throughput.
    Columns: input_ids, attention_mask, labels.
    """
    rows: list[dict] = []
    for ann in annotations:
        field_spans = [
            (f["start_char"], f["end_char"], f["label"]) for f in ann["fields"]
        ]
        windows = tokenize_document(
            ann["text"], tokenizer, email_spans=None, stride=_FIELD_STRIDE
        )
        for w in windows:
            labels = annotate_window(w["offset_mapping"], w["word_ids"], field_spans)
            rows.append(
                {
                    "input_ids": w["input_ids"],
                    "attention_mask": w["attention_mask"],
                    "labels": labels,
                }
            )
    return Dataset.from_list(rows)


def train(
    train_path: str | Path,
    dev_path: str | Path,
    output_dir: str | Path,
    model_name: str = PRIMARY_MODEL,
    num_epochs: int = 15,
    batch_size: int = 8,
    early_stopping_patience: int = 3,
    use_class_weights: bool = True,
) -> None:
    """
    Fine-tune RobBERT for email field extraction using Focal Loss.

    Parameters
    ----------
    train_path              : path to stage-2 train JSON
    dev_path                : path to stage-2 dev JSON
    output_dir              : where to save the final model and tokenizer
    model_name              : HuggingFace model ID to start from
    num_epochs              : maximum training epochs (early stopping may halt earlier)
    batch_size              : per-device batch size
    early_stopping_patience : stop if val loss does not improve for this many epochs
    use_class_weights       : if True, apply sqrt-inverse-frequency class weights as
                              the alpha parameter of Focal Loss (recommended); if False,
                              use plain Focal Loss without class balancing (baseline)
    """
    print("Loading model …  (first run downloads ~500 MB of weights)")
    model, tokenizer = load_model_and_tokenizer(model_name)

    print("Building datasets …")
    train_dataset = build_hf_dataset(load_annotations(train_path), tokenizer)
    dev_dataset = build_hf_dataset(load_annotations(dev_path), tokenizer)
    print(f"  train windows: {len(train_dataset)}  dev windows: {len(dev_dataset)}")

    if use_class_weights:
        class_weights = _compute_class_weights(train_dataset)
        print(f"  class weights — max: {class_weights.max():.2f}  "
              f"(B-ATTACHMENT: {class_weights[LABEL2ID['B-ATTACHMENT']]:.2f}, "
              f"B-CC: {class_weights[LABEL2ID['B-CC']]:.2f}, "
              f"B-SUBJECT: {class_weights[LABEL2ID['B-SUBJECT']]:.2f})")
    else:
        class_weights = None
        print("  class weights disabled — using plain Focal Loss (unweighted baseline)")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=10,
        report_to="none",
    )

    trainer = _FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    print("Training …")
    trainer.train()

    print(f"Saving model to {output_dir} …")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("Done.")
