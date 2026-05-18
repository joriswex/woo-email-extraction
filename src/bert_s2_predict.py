"""
Inference: load a trained field-extraction model and predict field spans in an email.

Email-type classification is rule-based (Option A): inspect the Onderwerp: line.
No separate classifier model is needed.
"""

from __future__ import annotations
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from bert_tokenize import tokenize_document, merge_window_predictions
from bert_s2_data import (
    token_labels_to_field_spans,
    classify_email_type,
)

_FIELD_STRIDE = 64


def predict_fields(
    text: str,
    model,
    tokenizer,
    max_length: int = 512,
    stride: int = _FIELD_STRIDE,
) -> list[dict]:
    """
    Run inference on an email text and return predicted field spans.

    Parameters
    ----------
    text      : individual email text (not the full dossier)
    model     : fine-tuned AutoModelForTokenClassification (in eval mode)
    tokenizer : the tokenizer used during training
    max_length, stride : must match training values

    Returns
    -------
    List of {"label": str, "start_char": int, "end_char": int} in document order.
    """
    model.eval()
    windows = tokenize_document(text, tokenizer, max_length=max_length, stride=stride)

    for win in windows:
        input_ids = torch.tensor([win["input_ids"]])
        attention_mask = torch.tensor([win["attention_mask"]])

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        win["predictions"] = logits[0].argmax(dim=-1).tolist()

    offsets, word_ids, label_ids = merge_window_predictions(windows, stride=stride)
    return token_labels_to_field_spans(offsets, word_ids, label_ids)


def predict_email(
    text: str,
    model,
    tokenizer,
    max_length: int = 512,
    stride: int = _FIELD_STRIDE,
) -> dict:
    """
    Predict both field spans and email type for a single email.

    Returns
    -------
    {
        "email_type": "NEW" | "REPLY" | "FORWARD",
        "fields": [{"label": str, "start_char": int, "end_char": int}, ...]
    }
    """
    return {
        "email_type": classify_email_type(text),
        "fields": predict_fields(text, model, tokenizer, max_length=max_length, stride=stride),
    }


def predict_email_from_checkpoint(
    text: str,
    checkpoint_dir: str | Path,
    max_length: int = 512,
    stride: int = _FIELD_STRIDE,
) -> dict:
    """
    Load a saved field-extraction checkpoint and predict fields + email type.

    Parameters
    ----------
    text           : individual email text
    checkpoint_dir : directory written by train_field.train()

    Returns
    -------
    Same as predict_email.
    """
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(checkpoint_dir))
    return predict_email(text, model, tokenizer, max_length=max_length, stride=stride)
