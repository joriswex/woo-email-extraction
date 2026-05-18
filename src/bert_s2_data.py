"""
Field-level annotation schema and BIO label conversion for individual emails.

Annotation schema (JSON) — email-level (stage 2)
-------------------------------------------------
{
    "dossier_id": "string",
    "email_id":   "string",
    "text":       "...",       # this email's text only; all offsets are relative to it
    "email_type": "NEW" | "REPLY" | "FORWARD",
    "fields": [
        {"label": "FROM",    "start_char": 5,   "end_char": 18},
        {"label": "TO",      "start_char": 24,  "end_char": 40},
        {"label": "DATE",    "start_char": 49,  "end_char": 59},
        {"label": "SUBJECT", "start_char": 71,  "end_char": 95},
        {"label": "BODY",    "start_char": 97,  "end_char": 310},
        {"label": "SIG",     "start_char": 312, "end_char": 370}
    ]
}

Field labels
------------
Each field value is BIO-encoded as B-<LABEL> / I-<LABEL>.  The header keywords
("Van:", "Aan:", "Datum:", "Onderwerp:") and surrounding whitespace are O.

REDACTED marks any field value that has been withheld — a Woo article code
like [5.1.2e], an OCR-empty region, or visually absent text.  The model learns:
"after Van:, the next span is either B-FROM or B-REDACTED."  REDACTED is a
value-state label, not a separate field; each span has exactly one label.

Email-type classification (rule-based, Option A)
------------------------------------------------
classify_email_type(text) inspects the "Onderwerp:" line:
    starts with "Re:"                          → REPLY
    starts with "Fw:", "Fwd:", "Doorsturen:"   → FORWARD
    otherwise (or no subject line found)       → NEW
"""

from __future__ import annotations
import re

# ---------------------------------------------------------------------------
# Label set
# ---------------------------------------------------------------------------

FIELD_TYPES = [
    "FROM", "TO", "CC", "DATE", "SUBJECT",
    "ATTACHMENT", "REDACTED",
]

LABEL2ID: dict[str, int] = {}
for _ft in FIELD_TYPES:
    LABEL2ID[f"B-{_ft}"] = len(LABEL2ID)
    LABEL2ID[f"I-{_ft}"] = len(LABEL2ID)
LABEL2ID["O"] = len(LABEL2ID)

ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}
IGNORED: int = -100

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_overlapping_field(
    tok_start: int,
    tok_end: int,
    field_spans: list[tuple[int, int, str]],
) -> tuple[int, str] | tuple[None, None]:
    """Return (index, label) of the first field overlapping [tok_start, tok_end)."""
    for i, (f_start, f_end, label) in enumerate(field_spans):
        if tok_start < f_end and tok_end > f_start:
            return i, label
    return None, None


# ---------------------------------------------------------------------------
# BIO annotation
# ---------------------------------------------------------------------------


def annotate_window(
    offset_mapping: list[tuple[int, int]],
    word_ids: list[int | None],
    field_spans: list[tuple[int, int, str]],
) -> list[int]:
    """
    Produce BIO field label IDs for one tokenizer window.

    Parameters
    ----------
    offset_mapping : character range (start, end) for each token in the window
    word_ids       : word index per token (None for special tokens / padding)
    field_spans    : list of (start_char, end_char, label) for each field

    Returns
    -------
    List of label IDs; IGNORED for special tokens and non-first sub-word pieces.

    Boundary rounding is outward (same as data.annotate_window): a token that
    overlaps a field span at all is included in that field.
    """
    labels: list[int] = []
    prev_field_idx: int | None = None
    prev_word_id: int | None = None

    for (tok_start, tok_end), wid in zip(offset_mapping, word_ids):
        if wid is None:
            labels.append(IGNORED)
            continue

        if wid == prev_word_id:
            labels.append(IGNORED)
            continue

        prev_word_id = wid
        field_idx, field_label = _find_overlapping_field(tok_start, tok_end, field_spans)

        if field_idx is None:
            labels.append(LABEL2ID["O"])
            prev_field_idx = None
        elif field_idx != prev_field_idx:
            labels.append(LABEL2ID[f"B-{field_label}"])
            prev_field_idx = field_idx
        else:
            labels.append(LABEL2ID[f"I-{field_label}"])

    return labels


def token_labels_to_field_spans(
    offset_mapping: list[tuple[int, int]],
    word_ids: list[int | None],
    label_ids: list[int],
) -> list[dict]:
    """
    Convert predicted BIO label IDs back to field span dicts.

    Mirrors the word-first filtering in annotate_window: only the first
    sub-token of each word is examined.

    Returns
    -------
    List of {"label": str, "start_char": int, "end_char": int} in document order.
    """
    spans: list[dict] = []
    current_label: str | None = None
    current_start: int | None = None
    current_end: int | None = None
    prev_word_id: int | None = None

    def _flush():
        if current_start is not None:
            spans.append(
                {"label": current_label, "start_char": current_start, "end_char": current_end}
            )

    for (tok_start, tok_end), wid, lid in zip(offset_mapping, word_ids, label_ids):
        if wid is None or wid == prev_word_id:
            continue
        prev_word_id = wid

        label_str = ID2LABEL.get(lid, "O")

        if label_str.startswith("B-"):
            _flush()
            current_label = label_str[2:]
            current_start = tok_start
            current_end = tok_end
        elif label_str.startswith("I-") and current_start is not None:
            current_end = tok_end
        else:
            _flush()
            current_label = None
            current_start = None
            current_end = None

    _flush()
    return spans


# ---------------------------------------------------------------------------
# Rule-based email-type classification
# ---------------------------------------------------------------------------

_SUBJECT_RE = re.compile(r"^Onderwerp:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
_REPLY_RE = re.compile(r"^Re\s*:", re.IGNORECASE)
_FORWARD_RE = re.compile(r"^(Fw\s*:|Fwd\s*:|Doorsturen\s*:)", re.IGNORECASE)


def classify_email_type(text: str) -> str:
    """
    Classify an email as NEW, REPLY, or FORWARD from the subject line.

    Searches for a "Onderwerp:" line and checks the subject value prefix.
    Returns NEW when no subject line is found.
    """
    m = _SUBJECT_RE.search(text)
    if m is None:
        return "NEW"
    subject = m.group(1).strip()
    if _REPLY_RE.match(subject):
        return "REPLY"
    if _FORWARD_RE.match(subject):
        return "FORWARD"
    return "NEW"
