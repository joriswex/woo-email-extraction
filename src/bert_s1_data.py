"""
Annotation schema and BIO label conversion functions.

Annotation schema (JSON)
------------------------
{
    "dossier_id": "string",
    "text": "...",     # full concatenated text of the dossier
    "emails": [
        {"email_id": "string", "start_char": 0, "end_char": 512},
        ...
    ]
}

NOTE: Embedding full text in JSON is convenient for small corpora but produces
large files for real dossiers (often >100 KB each). When annotating real data,
consider storing text in a separate file and referencing it by path.

Label IDs
---------
    B-EMAIL =  0   first word-token of an email
    I-EMAIL =  1   continuation word-token inside an email
    O       =  2   outside any email (cover pages, gaps, tables of contents)
    -100         ignored by the loss: special tokens (<s>, </s>, padding) and
                 non-first sub-word pieces within a word

Sub-word handling
-----------------
RobBERT uses BPE; one word may become several tokens. Only the FIRST sub-token
of each word carries a BIO label. Subsequent sub-tokens receive -100 and are
ignored by the CrossEntropyLoss. This is the standard HuggingFace NER approach.

Boundary rounding
-----------------
When an annotation boundary (start_char or end_char) falls in the middle of a
word, the token is included in the email if it OVERLAPS with the span at all:
    overlap ⟺ tok_start < e_end AND tok_end > e_start
This rounds outward: the email grows to cover the full word at each edge.
"""

from __future__ import annotations

LABEL2ID: dict[str, int] = {"B-EMAIL": 0, "I-EMAIL": 1, "O": 2}
ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}
IGNORED: int = -100


def _find_overlapping_email(
    tok_start: int,
    tok_end: int,
    email_spans: list[tuple[int, int]],
) -> int | None:
    """Return the index of the first email overlapping [tok_start, tok_end), or None."""
    for i, (e_start, e_end) in enumerate(email_spans):
        if tok_start < e_end and tok_end > e_start:
            return i
    return None


def annotate_window(
    offset_mapping: list[tuple[int, int]],
    word_ids: list[int | None],
    email_spans: list[tuple[int, int]],
) -> list[int]:
    """
    Produce BIO label IDs for one tokenizer window.

    Parameters
    ----------
    offset_mapping : character range (start, end) for each token in the window
    word_ids       : word index per token from tokenizer.word_ids(); None for
                     special tokens and padding
    email_spans    : list of (start_char, end_char) for each email in the dossier

    Returns
    -------
    List of label IDs, one per token position, with IGNORED (-100) for special
    tokens and non-first sub-word pieces.

    Adjacent emails produce ...I-EMAIL B-EMAIL... with no O gap between them.
    """
    labels: list[int] = []
    prev_email_idx: int | None = None
    prev_word_id: int | None = None

    for (tok_start, tok_end), wid in zip(offset_mapping, word_ids):
        if wid is None:
            labels.append(IGNORED)
            continue

        if wid == prev_word_id:
            labels.append(IGNORED)
            continue

        prev_word_id = wid
        email_idx = _find_overlapping_email(tok_start, tok_end, email_spans)

        if email_idx is None:
            label_id = LABEL2ID["O"]
            prev_email_idx = None
        elif email_idx != prev_email_idx:
            label_id = LABEL2ID["B-EMAIL"]
            prev_email_idx = email_idx
        else:
            label_id = LABEL2ID["I-EMAIL"]

        labels.append(label_id)

    return labels


def token_labels_to_char_spans(
    offset_mapping: list[tuple[int, int]],
    word_ids: list[int | None],
    label_ids: list[int],
) -> list[tuple[int, int]]:
    """
    Convert predicted BIO label IDs back to character-level email spans.

    Mirrors the word-first filtering in annotate_window: only the first
    sub-token of each word is examined. Returns [(start_char, end_char), ...]
    for each predicted email, in document order.
    """
    spans: list[tuple[int, int]] = []
    current_start: int | None = None
    current_end: int | None = None
    prev_word_id: int | None = None

    for (tok_start, tok_end), wid, label in zip(offset_mapping, word_ids, label_ids):
        if wid is None or wid == prev_word_id:
            continue
        prev_word_id = wid

        if label == LABEL2ID["B-EMAIL"]:
            if current_start is not None:
                spans.append((current_start, current_end))
            current_start = tok_start
            current_end = tok_end
        elif label == LABEL2ID["I-EMAIL"] and current_start is not None:
            current_end = tok_end
        else:
            if current_start is not None:
                spans.append((current_start, current_end))
            current_start = None
            current_end = None

    if current_start is not None:
        spans.append((current_start, current_end))

    return spans
