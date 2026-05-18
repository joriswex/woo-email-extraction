"""
Regex-based field extraction and email boundary detection.

Programmatic interface for evaluation — adapted from preannotate.py but
with several improvements that differentiate it as a stronger pipeline:

  1. Multi-line TO/CC: recipient lists spanning multiple lines are captured in full
  2. Two-line SUBJECT: subjects that wrap to a second line are captured
  3. Attachment filename scanning: filenames with extensions (.pdf, .docx etc.)
     appearing immediately after the subject line are detected even without
     a Bijlagen: keyword — matching how they appear as clickable links in PDFs
  4. Outermost-header deduplication: for fields that should appear once
     (FROM, TO, DATE, SUBJECT), only the first occurrence is kept, preventing
     forwarded email chains from contributing extra spurious field values

  extract_fields(email_text)        → field value dicts for one email
  detect_email_boundaries(text)     → (start_char, end_char) list for a dossier
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_FROM_RE       = re.compile(r'^(From|Van)\s*:',        re.IGNORECASE | re.MULTILINE)
_TO_RE         = re.compile(r'^(To|Aan)\s*:',          re.IGNORECASE | re.MULTILINE)
_CC_RE         = re.compile(r'^(Cc|CC)\s*:',           re.IGNORECASE | re.MULTILINE)
_DATE_RE       = re.compile(r'^(Sent|Date|Datum|Verzonden)\s*:', re.IGNORECASE | re.MULTILINE)
_SUBJECT_RE    = re.compile(r'^(Subject|Onderwerp)\s*:', re.IGNORECASE | re.MULTILINE)
_ATTACHMENT_RE = re.compile(r'^(Attachments?|Bijlagen?)\s*:', re.IGNORECASE | re.MULTILINE)

# File extensions that indicate an attachment filename in the PDF
_FILENAME_RE = re.compile(
    r'\S+\.(?:pdf|docx?|xlsx?|pptx?|msg|zip|txt|csv|eml|rtf)\b',
    re.IGNORECASE,
)

# Matches any header keyword at the start of a line — signals end of a field value
_HEADER_START_RE = re.compile(
    r'^(From|Van|To|Aan|Cc|CC|Sent|Date|Datum|Verzonden|Subject|Onderwerp|Attachments?|Bijlagen?)\s*:',
    re.IGNORECASE,
)

# Fields that should appear only once per email — first occurrence wins
# (prevents forwarded-chain headers from contributing extra values)
_UNIQUE_FIELDS = {"FROM", "TO", "DATE", "SUBJECT"}

# max_extra_lines: 0 = single line, 1 = up to one continuation line, None = unlimited
_FIELD_PATTERNS = [
    (_FROM_RE,       "FROM",       0),
    (_TO_RE,         "TO",         None),
    (_CC_RE,         "CC",         None),
    (_DATE_RE,       "DATE",       0),
    (_SUBJECT_RE,    "SUBJECT",    0),
    (_ATTACHMENT_RE, "ATTACHMENT", 0),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _value_span(
    text: str,
    m: re.Match,
    max_extra_lines: int | None = 0,
) -> tuple[int, int] | None:
    """
    Span of the field value starting after the header keyword colon.

    max_extra_lines=0    : single line only
    max_extra_lines=1    : up to one continuation line (e.g. long subjects)
    max_extra_lines=None : unlimited continuation until blank line or next header
    """
    colon = text.index(":", m.start())
    vs = colon + 1
    while vs < len(text) and text[vs] == " ":
        vs += 1

    ve = vs
    pos = vs
    extra_lines_collected = 0

    while pos < len(text):
        nl = text.find("\n", pos)
        line_end = len(text) if nl == -1 else nl

        candidate = line_end
        while candidate > vs and text[candidate - 1] in (" ", "\r"):
            candidate -= 1
        if candidate > vs:
            ve = candidate

        if nl == -1:
            break

        # Stop if we have collected enough lines
        if max_extra_lines is not None and extra_lines_collected >= max_extra_lines:
            break

        next_start = nl + 1
        if next_start >= len(text):
            break
        next_line = text[next_start:next_start + 60]

        if next_line.strip() == "" or _HEADER_START_RE.match(next_line):
            break

        extra_lines_collected += 1
        pos = next_start

    return (vs, ve) if ve > vs else None


def _scan_attachments_after_subject(
    email_text: str,
    subject_end: int,
    max_lines: int = 5,
) -> list[dict]:
    """
    Scan the lines immediately after the subject value for attachment filenames.

    In Woo PDF dossiers, attachments often appear as clickable filename links
    directly after the subject line without a Bijlagen: keyword. This scans
    up to max_lines lines (stopping at a blank line) and returns any found
    filenames as ATTACHMENT spans.
    """
    results = []
    pos = subject_end
    lines_checked = 0

    while pos < len(text := email_text) and lines_checked < max_lines:
        nl = text.find("\n", pos)
        line_end = len(text) if nl == -1 else nl
        line = text[pos:line_end].strip()

        if not line:
            break  # blank line = end of header area

        if _HEADER_START_RE.match(line):
            break  # new header keyword — stop

        for match in _FILENAME_RE.finditer(text, pos, line_end):
            results.append({
                "label": "ATTACHMENT",
                "value": match.group().strip(),
                "start_char": match.start(),
                "end_char": match.end(),
            })

        if nl == -1:
            break
        pos = nl + 1
        lines_checked += 1

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_fields(email_text: str) -> list[dict]:
    """
    Extract field values from a single email text using regex.

    Improvements over the pre-annotation pipeline:
    - TO/CC: full multi-line recipient lists captured
    - SUBJECT: up to one continuation line captured
    - ATTACHMENT: filenames with extensions found after the subject line
      even without a Bijlagen: keyword
    - Unique fields (FROM, TO, DATE, SUBJECT): only first occurrence kept
      to avoid picking up forwarded-chain headers

    Returns
    -------
    List of {"label": str, "value": str, "start_char": int, "end_char": int}
    sorted by start position.
    """
    spans = []
    seen_unique: set[str] = set()
    subject_end: int | None = None

    for pattern, label, max_extra in _FIELD_PATTERNS:
        for m in pattern.finditer(email_text):
            # For unique fields, skip if we already have one
            if label in _UNIQUE_FIELDS and label in seen_unique:
                continue

            span = _value_span(email_text, m, max_extra_lines=max_extra)
            if span:
                s, e = span
                spans.append({
                    "label": label,
                    "value": email_text[s:e].strip(),
                    "start_char": s,
                    "end_char": e,
                })
                if label in _UNIQUE_FIELDS:
                    seen_unique.add(label)
                if label == "SUBJECT":
                    subject_end = e  # record where subject ends for attachment scan

    # Scan for attachment filenames after the subject line
    if subject_end is not None:
        for att in _scan_attachments_after_subject(email_text, subject_end):
            # Only add if not already captured by a Bijlagen: keyword
            if not any(
                s["label"] == "ATTACHMENT" and
                abs(s["start_char"] - att["start_char"]) < 10
                for s in spans
            ):
                spans.append(att)

    return sorted(spans, key=lambda x: x["start_char"])


def detect_email_boundaries(text: str) -> list[tuple[int, int]]:
    """
    Detect email boundaries in a dossier text.

    Primary anchors  : Van: / From: (sender fields — first header line)
    Fallback anchors : To: / Aan: (recipient fields — used only when no sender
                       line is found within 300 chars before them)

    Van: and From: are treated identically as Dutch/English sender equivalents.
    An anchor qualifies only when Subject:/Onderwerp: appears within 500 chars.

    Returns [(start_char, end_char)] in document order.
    """
    starts: list[int] = []

    # Primary: sender lines — Van: (Dutch) and From: (English)
    for m in re.finditer(r'^(?:Van|From)\s*:', text, re.IGNORECASE | re.MULTILINE):
        window = text[m.start(): m.start() + 500]
        if re.search(r'^(?:Subject|Onderwerp)\s*:', window, re.IGNORECASE | re.MULTILINE):
            starts.append(m.start())

    # Fallback: recipient lines — only when no sender line precedes within 300 chars
    for m in re.finditer(r'^(?:To|Aan)\s*:', text, re.IGNORECASE | re.MULTILINE):
        window = text[m.start(): m.start() + 500]
        if not re.search(r'^(?:Subject|Onderwerp)\s*:', window, re.IGNORECASE | re.MULTILINE):
            continue
        preceding = text[max(0, m.start() - 300): m.start()]
        if re.search(r'^(?:Van|From)\s*:', preceding, re.IGNORECASE | re.MULTILINE):
            continue
        starts.append(m.start())

    starts.sort()
    if not starts:
        return []

    # Deduplicate starts within 20 chars
    deduped = [starts[0]]
    for s in starts[1:]:
        if s - deduped[-1] > 20:
            deduped.append(s)
    starts = deduped

    spans = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        while end > start and text[end - 1] in (" ", "\n", "\r"):
            end -= 1
        if end > start:
            spans.append((start, end))
    return spans
