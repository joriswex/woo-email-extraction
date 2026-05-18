"""
Inference for line-level email boundary detection.

Loads a fine-tuned AutoModelForSequenceClassification checkpoint and
predicts which lines in a dossier text are email starts, then converts
those line positions into (start_char, end_char) email spans.
"""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bert_s1_train import CONTEXT_LINES, MAX_LENGTH, LABEL_START


def predict(
    dossier_text: str,
    model,
    tokenizer,
    threshold: float = 0.5,
) -> list[tuple[int, int]]:
    """
    Predict email boundary spans in a dossier text using line-level classification.

    Parameters
    ----------
    dossier_text : full extracted text of one dossier
    model        : fine-tuned AutoModelForSequenceClassification
    tokenizer    : matching tokenizer
    threshold    : probability threshold for classifying a line as an email start

    Returns
    -------
    [(start_char, end_char)] in document order
    """
    model.eval()
    lines = dossier_text.split("\n")

    # Build char offset for each line
    offsets: list[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1

    email_starts: list[int] = []

    for i, (line, char_start) in enumerate(zip(lines, offsets)):
        if not line.strip():
            continue

        ctx_start = max(0, i - CONTEXT_LINES)
        ctx_end   = min(len(lines), i + CONTEXT_LINES + 1)
        context   = "\n".join(lines[ctx_start:ctx_end])

        encoded = tokenizer(
            context,
            max_length=MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**encoded).logits
        prob_start = torch.softmax(logits, dim=-1)[0, LABEL_START].item()

        if prob_start >= threshold:
            email_starts.append(char_start)

    if not email_starts:
        return []

    # Build spans: each email runs from its start to the next
    spans = []
    for i, start in enumerate(email_starts):
        end = email_starts[i + 1] if i + 1 < len(email_starts) else len(dossier_text)
        while end > start and dossier_text[end - 1] in (" ", "\n", "\r"):
            end -= 1
        if end > start:
            spans.append((start, end))
    return spans


def merge_nearby_starts(
    spans: list[tuple[int, int]],
    text: str,
    max_lines: int = 8,
) -> list[tuple[int, int]]:
    """
    Post-processing filter: collapse consecutive predicted email starts that
    fall within max_lines of each other into just the first one.

    This removes false positives caused by the model firing on multiple lines
    within the same header block (e.g. both Van: and Verzonden:).
    """
    if not spans:
        return spans

    filtered = [spans[0]]
    for start, end in spans[1:]:
        prev_start = filtered[-1][0]
        lines_between = text[prev_start:start].count("\n")
        if lines_between > max_lines:
            filtered.append((start, end))
        else:
            # Extend the previous span to cover up to this one's end
            filtered[-1] = (filtered[-1][0], end)
    return filtered


def predict_from_checkpoint(
    dossier_text: str,
    checkpoint_dir: str | Path,
    threshold: float = 0.5,
) -> list[tuple[int, int]]:
    """Load checkpoint and predict email spans."""
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    model     = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir))
    return predict(dossier_text, model, tokenizer, threshold=threshold)
