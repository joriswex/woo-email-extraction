"""
Regex-based pre-annotation for Label Studio import tasks.

Detects email boundaries and field values using header-line patterns and
injects them as Label Studio predictions. When you open a task you will see
emails and fields already highlighted — just correct mistakes and confirm
what is right.

Detects both English (To:/From:/Sent:/Subject:) and Dutch
(Van:/Aan:/Datum:/Verzonden:/Onderwerp:) email formats.

Usage
-----
    # Pre-annotate one dossier
    python scripts/preannotate.py data/raw/21390238.pdf

    # All dossiers at once
    python scripts/preannotate.py data/raw/*.pdf

    # Custom output directory
    python scripts/preannotate.py data/raw/*.pdf --output_dir data/import/

Output
------
One JSON file per PDF in data/import/, ready to upload to Label Studio.
"""

from __future__ import annotations
import argparse
import json
import re
import sys
import uuid
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from pdf_extract import extract_text

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Keywords that start a header line
_TO_RE         = re.compile(r'^(To|Aan)\s*:', re.IGNORECASE | re.MULTILINE)
_FROM_RE       = re.compile(r'^(From|Van)\s*:', re.IGNORECASE | re.MULTILINE)
_CC_RE         = re.compile(r'^(Cc|CC)\s*:', re.IGNORECASE | re.MULTILINE)
_DATE_RE       = re.compile(r'^(Sent|Datum|Verzonden)\s*:', re.IGNORECASE | re.MULTILINE)
_SUBJECT_RE    = re.compile(r'^(Subject|Onderwerp)\s*:', re.IGNORECASE | re.MULTILINE)
_ATTACHMENT_RE = re.compile(r'^(Attachments?|Bijlagen?)\s*:', re.IGNORECASE | re.MULTILINE)

# An email start: a To: or Van: line followed within 400 chars by a Subject:/Onderwerp: line
_EMAIL_START_RE = re.compile(
    r'^(?:To|Aan|Van)\s*:.+?(?=^(?:Subject|Onderwerp)\s*:)',
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_HEADER_START_RE = re.compile(
    r'^(From|Van|To|Aan|Cc|CC|Sent|Date|Datum|Verzonden|Subject|Onderwerp|Attachments?|Bijlagen?)\s*:',
    re.IGNORECASE,
)


def _value_span(text: str, keyword_match: re.Match, multiline: bool = False) -> tuple[int, int] | None:
    """Return (start, end) of the field value after the keyword colon.
    If multiline=True, collects continuation lines until a blank line or
    new header keyword (used for TO and CC recipient lists).
    """
    colon_pos = text.index(':', keyword_match.start())
    vs = colon_pos + 1
    while vs < len(text) and text[vs] == ' ':
        vs += 1

    if not multiline:
        line_end = text.find('\n', vs)
        ve = len(text) if line_end == -1 else line_end
        while ve > vs and text[ve - 1] in (' ', '\r'):
            ve -= 1
        return (vs, ve) if ve > vs else None

    ve = vs
    pos = vs
    while pos < len(text):
        nl = text.find('\n', pos)
        line_end = len(text) if nl == -1 else nl

        candidate = line_end
        while candidate > vs and text[candidate - 1] in (' ', '\r'):
            candidate -= 1
        if candidate > vs:
            ve = candidate

        if nl == -1:
            break

        next_start = nl + 1
        if next_start >= len(text):
            break
        next_line = text[next_start:next_start + 60]

        if next_line.strip() == '' or _HEADER_START_RE.match(next_line):
            break

        pos = next_start

    return (vs, ve) if ve > vs else None


def _find_email_boundaries(text: str) -> list[tuple[int, int]]:
    """
    Find email boundary spans.

    Primary anchors  : Van: / From: (sender fields — Dutch/English equivalents)
    Fallback anchors : To: / Aan: (recipient fields — used only when no sender
                       line appears within 300 chars before them)

    An anchor qualifies only when Subject:/Onderwerp: appears within 500 chars.
    """
    starts = []

    # Primary: sender lines
    for m in re.finditer(r'^(?:Van|From)\s*:', text, re.IGNORECASE | re.MULTILINE):
        window = text[m.start(): m.start() + 500]
        if re.search(r'^(?:Subject|Onderwerp)\s*:', window, re.IGNORECASE | re.MULTILINE):
            starts.append(m.start())

    # Fallback: recipient lines only when no sender line precedes within 300 chars
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

    # Deduplicate starts that are very close together (within 20 chars)
    deduped = [starts[0]]
    for s in starts[1:]:
        if s - deduped[-1] > 20:
            deduped.append(s)
    starts = deduped

    # Build spans: each email runs from its start to the start of the next
    spans = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        # Trim trailing whitespace/newlines
        while end > start and text[end - 1] in (' ', '\n', '\r'):
            end -= 1
        if end > start:
            spans.append((start, end))

    return spans


def _build_predictions(text: str, dossier_id: str, email_only: bool = False) -> list[dict]:
    """Build Label Studio prediction result items for one dossier text."""
    result = []
    email_spans = _find_email_boundaries(text)

    for email_idx, (e_start, e_end) in enumerate(email_spans):
        email_text = text[e_start:e_end]

        # EMAIL boundary
        result.append({
            "id": str(uuid.uuid4())[:8],
            "type": "labels",
            "from_name": "label",
            "to_name": "text",
            "value": {"start": e_start, "end": e_end, "labels": ["EMAIL"]},
        })

        if email_only:
            continue

        # Field values within this email
        field_patterns = [
            (_FROM_RE,       "FROM",       False),
            (_TO_RE,         "TO",         True),
            (_CC_RE,         "CC",         True),
            (_DATE_RE,       "DATE",       False),
            (_SUBJECT_RE,    "SUBJECT",    False),
            (_ATTACHMENT_RE, "ATTACHMENT", False),
        ]
        for pattern, label, multiline in field_patterns:
            for m in pattern.finditer(email_text):
                span = _value_span(email_text, m, multiline=multiline)
                if span:
                    v_start, v_end = span
                    if v_end - v_start > 1:
                        result.append({
                            "id": str(uuid.uuid4())[:8],
                            "type": "labels",
                            "from_name": "label",
                            "to_name": "text",
                            "value": {
                                "start": e_start + v_start,
                                "end":   e_start + v_end,
                                "labels": [label],
                            },
                        })

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def prepare(pdf_path: Path, output_dir: Path, email_only: bool = False) -> None:
    dossier_id = pdf_path.stem
    # Always look for cached text in data/import/ (full pre-annotations),
    # regardless of where the output is being written
    cached = _ROOT / "data" / "import" / f"{dossier_id}.json"
    if not cached.exists():
        cached = output_dir / f"{dossier_id}.json"

    # Reuse text from existing import file to avoid re-running OCR
    if cached.exists():
        with open(cached, encoding="utf-8") as f:
            existing = json.load(f)
        text = existing[0]["data"]["text"]
        print(f"Processing {pdf_path.name} … (text from cache)", end=" ", flush=True)
        page_map = {}
    else:
        print(f"Processing {pdf_path.name} …", end=" ", flush=True)
        text, page_map = extract_text(pdf_path)

    predictions = _build_predictions(text, dossier_id, email_only=email_only)
    n_emails = sum(1 for r in predictions if r["value"]["labels"] == ["EMAIL"])
    mode = "EMAIL spans only" if email_only else "EMAIL + fields"
    print(f"{len(page_map)} pages  →  {n_emails} emails pre-annotated ({mode})")

    task = [{
        "data": {"text": text, "dossier_id": dossier_id},
        "predictions": [{
            "model_version": "regex-v1",
            "score": 0.7,
            "result": predictions,
        }],
    }]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{dossier_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(task, f, ensure_ascii=False, indent=2)

    print(f"  → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-annotate Woo dossier PDFs for Label Studio using regex."
    )
    parser.add_argument("pdfs", nargs="+", help="PDF file(s) to process")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to write import JSON files. Defaults to data/import/stage1/ "
             "when --email-only is set, data/import/ otherwise.",
    )
    parser.add_argument(
        "--email-only",
        action="store_true",
        help="Only pre-annotate EMAIL boundaries (for Stage 1). "
             "Writes to data/import/stage1/ by default.",
    )
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.email_only:
        output_dir = _ROOT / "data" / "import" / "stage1"
    else:
        output_dir = _ROOT / "data" / "import"

    for pdf in args.pdfs:
        prepare(Path(pdf), output_dir, email_only=args.email_only)

    print(f"\nDone. Upload file(s) from {output_dir} into Label Studio.")


if __name__ == "__main__":
    main()
