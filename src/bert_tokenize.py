"""
Sliding-window tokenisation for long documents.

RobBERT (RoBERTa-family) has a maximum context of 512 tokens. Documents longer
than ~450 words are split into overlapping windows; predictions are merged at
inference time by discarding edge tokens from the overlap region.

Window parameters
-----------------
    max_length = 512   tokens per window (hard limit of the model)
    stride     = 128   tokens of overlap between consecutive windows
    step       = 384   non-overlapping tokens advanced per window

Trusted zone at inference
--------------------------
Each overlapping edge contributes stride // 2 = 64 tokens of uncertainty.
The trusted zone of window i is:
    [64, 448)   for interior windows
    [0,  448)   for the first window
    [64, end)   for the last window
Tokens outside the trusted zone are discarded when merging predictions.
"""

from __future__ import annotations
from bert_s1_data import annotate_window, LABEL2ID, IGNORED

_TRUST_MARGIN = 64  # = stride // 2 for the default stride of 128


def tokenize_document(
    text: str,
    tokenizer,
    email_spans: list[tuple[int, int]] | None = None,
    max_length: int = 512,
    stride: int = 128,
) -> list[dict]:
    """
    Tokenise a document into overlapping windows for training or inference.

    Each window dict contains:
        input_ids       : list[int]
        attention_mask  : list[int]
        offset_mapping  : list[tuple[int, int]]   char range per token
        word_ids        : list[int | None]         word index per token
        window_idx      : int                      0-based index of this window
        n_windows       : int                      total number of windows
        labels          : list[int]                only present if email_spans given

    At training time, windows from the same document are independent examples.
    A window that starts in the middle of an email uses true labels (I-EMAIL
    for continuation tokens), so the model sees the correct distribution even
    without prior-window context.

    Parameters
    ----------
    text        : full document text
    tokenizer   : a HuggingFace *fast* tokenizer (required for offset_mapping)
    email_spans : [(start_char, end_char), ...]; omit for inference-only use
    max_length  : tokens per window
    stride      : overlap in tokens between consecutive windows
    """
    encoding = tokenizer(
        text,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        truncation=True,
        padding=False,
    )

    n_windows = len(encoding["input_ids"])
    windows = []

    for i in range(n_windows):
        word_ids = encoding.word_ids(batch_index=i)
        offset_mapping = encoding["offset_mapping"][i]

        window: dict = {
            "input_ids": encoding["input_ids"][i],
            "attention_mask": encoding["attention_mask"][i],
            "offset_mapping": offset_mapping,
            "word_ids": word_ids,
            "window_idx": i,
            "n_windows": n_windows,
        }

        if email_spans is not None:
            window["labels"] = annotate_window(offset_mapping, word_ids, email_spans)

        windows.append(window)

    return windows


def merge_window_predictions(
    windows: list[dict],
    stride: int = 128,
) -> tuple[list[tuple[int, int]], list[int | None], list[int]]:
    """
    Merge per-token predictions from overlapping windows into a flat sequence.

    For each unique character position, takes the prediction from the window
    where that position falls in the trusted zone. Positions are deduplicated
    by their (start_char, end_char) pair; the trusted-zone token wins.

    Returns
    -------
    offset_mapping : character ranges for the merged token sequence
    word_ids       : word indices for the merged sequence
    label_ids      : predicted label per merged token
    """
    margin = stride // 2
    # (start_char, end_char) -> (word_id, label)
    position_data: dict[tuple[int, int], tuple[int | None, int]] = {}

    for win in windows:
        i = win["window_idx"]
        n = win["n_windows"]
        word_ids = win["word_ids"]
        offset_mapping = win["offset_mapping"]
        predictions = win["predictions"]

        trusted_start = margin if i > 0 else 0
        trusted_end = len(word_ids) - margin if i < n - 1 else len(word_ids)

        for j, (wid, offset, label) in enumerate(
            zip(word_ids, offset_mapping, predictions)
        ):
            if wid is None:
                continue  # skip special tokens and padding
            if j < trusted_start or j >= trusted_end:
                continue  # outside trusted zone; another window covers this

            char_key = tuple(offset)
            if char_key not in position_data:
                position_data[char_key] = (wid, label)

    sorted_offsets = sorted(position_data.keys(), key=lambda x: x[0])
    word_ids_out = [position_data[k][0] for k in sorted_offsets]
    labels_out = [position_data[k][1] for k in sorted_offsets]

    return list(sorted_offsets), word_ids_out, labels_out
