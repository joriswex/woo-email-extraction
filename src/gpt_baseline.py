"""
GPT-4o baseline for field extraction and email boundary detection.

Stage 2 — field extraction:
  extract_fields_vision()         GPT-4o vision: PDF page images for one email

Stage 1 — boundary detection:
  detect_email_boundaries_text()  GPT-4o text: full dossier text, returns starts

Requires OPENAI_API_KEY to be set in the environment.
"""
from __future__ import annotations

import base64
import json
import re
from io import BytesIO
from pathlib import Path

from openai import OpenAI

_client: OpenAI | None = None
_MODEL = "gpt-5.5-2026-04-23"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_STAGE2_PROMPT_V3 = """\
You are extracting metadata from a single email in a Dutch government Woo dossier.

IMPORTANT CONTEXT: These documents contain two types of difficult content:
1. Heavy redaction — names replaced with codes like "5.1.2e", "BS/AL/DGB/...",
   greyed-out boxes, or strings of underscores/symbols.
2. OCR artefacts — wrongly captured letters, tildes (~), underscores (____),
   special characters (®, =), broken timestamps (e.g. "202412:22"), or lowercase
   header keywords (e.g. "van:" instead of "Van:").

CRITICAL RULE — you must ALWAYS return a value for every header field that is
present in the email, even if the value looks completely garbled:
- A field is ABSENT only if its header keyword does not appear at all
  (e.g. no "Van:", "From:", "Aan:", "Verzonden:" etc. anywhere in the email).
- If the header keyword IS present but the value is garbled, corrupted, or
  consists entirely of symbols — copy it EXACTLY as it appears. Never return
  null for a field whose keyword is visible. It is always better to return a
  garbled value than to return null.

The extracted text of the specific email is provided after the image(s).
Locate the correct email and copy all field values verbatim from that text.

Extract these fields:
- FROM: sender after Van: or From:
- TO: recipient after Aan: or To:
- CC: CC recipients, may be multiple
- DATE: date/time after Verzonden:, Datum:, Sent:, or Date:
- SUBJECT: subject after Onderwerp: or Subject:
- ATTACHMENT: filenames after Bijlagen: or Attachments:

Return ONLY a JSON object — no explanation:
{
  "FROM": "value or null",
  "TO": "value or null",
  "CC": ["value1", "value2"],
  "DATE": "value or null",
  "SUBJECT": "value or null",
  "ATTACHMENT": ["filename1", "filename2"]
}
null ONLY if the header keyword is completely absent. [] for absent list fields.\
"""

_STAGE2_PROMPT_V2 = """\
You are extracting metadata from a single email in a Dutch government Woo dossier.

IMPORTANT CONTEXT: These documents contain two types of difficult content that you
must handle without omitting values:
1. Heavy redaction — personal names and email addresses are replaced with anonymisation
   codes such as "5.1.2e", "5.1.2.e", "BS/AL/DGB/...", or shown as greyed-out boxes
   in the image. Copy these codes exactly as they appear; do not skip the field.
2. OCR artefacts — the text layer may contain wrongly captured letters, strange symbols,
   broken words, or garbled characters (e.g. "s.1.~", "@belas~t_ni~d_ie_n_s_tn._l~").
   Copy the value exactly as it appears in the extracted text — do not correct or clean it.

The PDF page image(s) show part of a long document with many emails concatenated.
The extracted text of the specific email you must process is provided after the image(s).
Use it to locate the correct email and copy all field values EXACTLY as they appear
in that extracted text, artefacts and redaction codes included.

Extract these fields from that specific email only:
- FROM: sender name and/or email address
- TO: recipient name(s) and/or email address(es)
- CC: CC recipient(s), may be multiple
- DATE: date and time the email was sent
- SUBJECT: email subject line
- ATTACHMENT: attachment filename(s), may be multiple

Return ONLY a JSON object with this exact structure — no explanation:
{
  "FROM": "value or null",
  "TO": "value or null",
  "CC": ["value1", "value2"],
  "DATE": "value or null",
  "SUBJECT": "value or null",
  "ATTACHMENT": ["filename1", "filename2"]
}
Use null for absent scalar fields and [] for absent list fields.\
"""

_STAGE2_PROMPT = """\
You are extracting metadata from a single email in a Dutch government Woo dossier.

The PDF page image(s) show part of a long document containing many individual emails \
concatenated one after another. Each email starts with a header block containing \
fields such as Van/From, Aan/To, CC, Verzonden/Datum/Sent/Date, Onderwerp/Subject, \
and optionally Bijlagen/Attachments.

The extracted text of the specific email you must process is provided after the \
image(s). Use it to locate the correct email in the image and copy all field values \
EXACTLY as they appear in that extracted text — including any redaction placeholders \
such as "5.1.2e", "[5.1.2e]", or similar codes. Do not guess or reconstruct \
redacted values.

Extract these fields from that specific email only:
- FROM: sender name and/or email address
- TO: recipient name(s) and/or email address(es)
- CC: CC recipient(s), may be multiple
- DATE: date and time the email was sent
- SUBJECT: email subject line
- ATTACHMENT: attachment filename(s), may be multiple

Return ONLY a JSON object with this exact structure — no explanation:
{
  "FROM": "value or null",
  "TO": "value or null",
  "CC": ["value1", "value2"],
  "DATE": "value or null",
  "SUBJECT": "value or null",
  "ATTACHMENT": ["filename1", "filename2"]
}
Use null for absent scalar fields and [] for absent list fields.\
"""

_STAGE1_PROMPT = """\
You are analyzing a Dutch government document (Woo dossier) that contains multiple \
email messages concatenated together.

Identify every individual email in the text. For each email, copy the exact text of \
its date/time line as it appears in the document. The date line usually starts with \
"Verzonden:", "Datum:", "Sent:", or "Date:" and includes the full date and time \
(e.g. "Verzonden: maandag 3 april 2023 14:35" or "Sent: Mon 4/3/2023 2:35:00 PM").
Dates are never redacted and are unique per email, making them reliable anchors.

Return ONLY a JSON array of strings — one entry per email, in document order. Example:
["Verzonden: maandag 3 april 2023 14:35", "Sent: Tue 4/4/2023 7:23:12 AM", "Datum: 5 mei 2023 09:15"]

Document text:
"""

_STAGE1_PROMPT_VISION = """\
You are analyzing a Dutch government document (Woo dossier) that contains multiple \
email messages concatenated together. The PDF page images of the full document are \
provided, followed by the extracted document text.

Use the PAGE IMAGES as your primary source to visually identify where each individual \
email starts. Look for email header blocks in the images — these contain fields such as \
Van:/From:, Verzonden:/Sent:, Aan:/To:, Onderwerp:/Subject: and visually stand out \
from the email body. The extracted text is provided only as a reference for copying \
date strings exactly.

For each email you identify from the images, return the exact text of its date/time \
line as it appears in the extracted text below the images \
(the line starting with "Verzonden:", "Datum:", "Sent:", or "Date:"). \
Dates are never redacted and are unique per email, making them reliable anchors.

Return ONLY a JSON array of strings — one entry per email, in document order. Example:
["Verzonden: maandag 3 april 2023 14:35", "Sent: Tue 4/4/2023 7:23:12 AM"]

The extracted document text follows the images.
"""

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _render_pages_b64(pdf_path: Path, page_numbers: list[int], resolution: int = 150) -> list[str]:
    """Render PDF pages (1-indexed) as base64-encoded PNG strings."""
    import pdfplumber
    result = []
    with pdfplumber.open(pdf_path) as pdf:
        for pn in page_numbers:
            img = pdf.pages[pn - 1].to_image(resolution=resolution)
            buf = BytesIO()
            img.save(buf, format="PNG")
            result.append(base64.b64encode(buf.getvalue()).decode())
    return result


def _pages_for_span(
    page_map: dict[int, tuple[int, int]],
    start: int,
    end: int,
) -> list[int]:
    """Return 1-indexed page numbers whose text range overlaps [start, end)."""
    return sorted(pn for pn, (ps, pe) in page_map.items() if ps < end and pe > start)


def _strip_markdown(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_fields_vision(
    pdf_path: Path,
    page_map: dict[int, tuple[int, int]],
    email_start_char: int,
    email_end_char: int,
    email_text: str = "",
    prompt_version: int = 1,
) -> list[dict]:
    """
    Call GPT-4o with PDF page image(s) + extracted email text for one email.

    Parameters
    ----------
    pdf_path        : dossier PDF file
    page_map        : {page_num: (start_char, end_char)} from pdf_extract
    email_start/end : character range of the email in the dossier text
    email_text      : pdfplumber-extracted text of this email (appended after images)

    Returns
    -------
    [{"label": str, "value": str}]  — empty list on parse failure
    """
    pages = _pages_for_span(page_map, email_start_char, email_end_char)
    if not pages:
        return []

    images_b64 = _render_pages_b64(pdf_path, pages)

    if prompt_version == 3:
        prompt = _STAGE2_PROMPT_V3
    elif prompt_version == 2:
        prompt = _STAGE2_PROMPT_V2
    else:
        prompt = _STAGE2_PROMPT
    content: list[dict] = [{"type": "text", "text": prompt}]
    for img_b64 in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "high"},
        })

    if email_text:
        content.append({
            "type": "text",
            "text": f"\n\nExtracted email text (copy field values exactly as they appear here):\n\n{email_text}",
        })

    resp = _get_client().chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": content}],
        max_completion_tokens=2048,  # GPT-5.5 uses reasoning tokens before visible output
    )
    raw = _strip_markdown(resp.choices[0].message.content)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    LIST_FIELDS = {"CC", "ATTACHMENT"}
    results = []
    for label in ("FROM", "TO", "CC", "DATE", "SUBJECT", "ATTACHMENT"):
        val = data.get(label)
        if label in LIST_FIELDS:
            items = val if isinstance(val, list) else ([val] if val else [])
            for v in items:
                if v:
                    results.append({"label": label, "value": str(v).strip()})
        else:
            if val:
                results.append({"label": label, "value": str(val).strip()})
    return results


def detect_email_boundaries_vision(
    pdf_path: Path,
    page_map: dict[int, tuple[int, int]],
    dossier_text: str,
) -> list[tuple[int, int]]:
    """
    GPT-5.5 vision + text Stage 1: send all PDF pages as images followed by
    the extracted dossier text.

    Images are the primary source for identifying email boundaries visually.
    The extracted text is appended so GPT can copy date strings exactly,
    ensuring returned strings match pdfplumber extraction for reliable anchoring.
    Same date-line matching logic as the text-only approach.
    """
    all_pages = sorted(page_map.keys())
    if not all_pages:
        return []

    # 72 DPI keeps the full dossier well under OpenAI's 50 MB image limit
    # while remaining sufficient for GPT to identify email header blocks.
    images_b64 = _render_pages_b64(pdf_path, all_pages, resolution=72)

    content: list[dict] = [{"type": "text", "text": _STAGE1_PROMPT_VISION}]
    for img_b64 in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "high"},
        })
    content.append({"type": "text", "text": f"\n\nExtracted document text:\n\n{dossier_text}"})

    resp = _get_client().chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": content}],
        max_completion_tokens=8192,
    )
    raw = _strip_markdown(resp.choices[0].message.content)

    try:
        first_lines: list[str] = json.loads(raw)
    except json.JSONDecodeError:
        return []

    # Same anchoring logic as detect_email_boundaries_text
    _HEADER_RE = re.compile(r'^(?:Van|From|To|Aan)\s*:', re.IGNORECASE | re.MULTILINE)

    starts: list[int] = []
    search_from = 0
    for line in first_lines:
        line = line.strip()
        if not line:
            continue
        idx = dossier_text.find(line, search_from)
        if idx < 0:
            continue
        preceding = dossier_text[max(0, idx - 500):idx]
        headers   = list(_HEADER_RE.finditer(preceding))
        if headers:
            email_start = max(0, idx - 500) + headers[-1].start()
        else:
            email_start = idx
        starts.append(email_start)
        search_from = idx + 1

    if not starts:
        return []

    spans = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(dossier_text)
        while e > s and dossier_text[e - 1] in (" ", "\n", "\r"):
            e -= 1
        if e > s:
            spans.append((s, e))
    return spans


def extract_fields_text_only(
    email_text: str,
    prompt_version: int = 3,
) -> list[dict]:
    """
    GPT-5.5 text-only Stage 2: send extracted email text without any images.

    Uses the same v3 prompt and JSON output format as extract_fields_vision
    so results are directly comparable. The absence of images is the only
    difference — this call tests what GPT can do from text alone.
    """
    if prompt_version == 3:
        prompt = _STAGE2_PROMPT_V3
    elif prompt_version == 2:
        prompt = _STAGE2_PROMPT_V2
    else:
        prompt = _STAGE2_PROMPT

    content: list[dict] = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": f"\n\nExtracted email text:\n\n{email_text}"},
    ]

    resp = _get_client().chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": content}],
        max_completion_tokens=2048,
    )
    raw = _strip_markdown(resp.choices[0].message.content)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    LIST_FIELDS = {"CC", "ATTACHMENT"}
    results = []
    for label in ("FROM", "TO", "CC", "DATE", "SUBJECT", "ATTACHMENT"):
        val = data.get(label)
        if label in LIST_FIELDS:
            items = val if isinstance(val, list) else ([val] if val else [])
            for v in items:
                if v:
                    results.append({"label": label, "value": str(v).strip()})
        else:
            if val:
                results.append({"label": label, "value": str(val).strip()})
    return results


def detect_email_boundaries_text(dossier_text: str) -> list[tuple[int, int]]:
    """
    Call GPT-4o (text mode) to identify email boundaries in a dossier.

    Sends the full dossier text and asks GPT-4o to return the first line of
    each email. Those lines are located in the text to derive char spans.

    Returns [(start_char, end_char)] in document order.
    """
    prompt = _STAGE1_PROMPT + dossier_text

    resp = _get_client().chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=8192,  # needs to fit first lines for 100+ emails
    )
    raw = _strip_markdown(resp.choices[0].message.content)

    try:
        first_lines: list[str] = json.loads(raw)
    except json.JSONDecodeError:
        return []

    # Locate each date line in the dossier text, then search backwards
    # for the nearest sender/recipient header line — that is the email start.
    _HEADER_RE = re.compile(r'^(?:Van|From|To|Aan)\s*:', re.IGNORECASE | re.MULTILINE)

    starts: list[int] = []
    search_from = 0
    for line in first_lines:
        line = line.strip()
        if not line:
            continue
        idx = dossier_text.find(line, search_from)
        if idx < 0:
            continue
        # Search backwards from date line for nearest header keyword
        preceding = dossier_text[max(0, idx - 500):idx]
        headers = list(_HEADER_RE.finditer(preceding))
        if headers:
            # Last header match before the date = email start
            email_start = max(0, idx - 500) + headers[-1].start()
        else:
            email_start = idx  # fallback: use date position itself
        starts.append(email_start)
        search_from = idx + 1  # enforce document order

    if not starts:
        return []

    spans = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(dossier_text)
        while e > s and dossier_text[e - 1] in (" ", "\n", "\r"):
            e -= 1
        if e > s:
            spans.append((s, e))
    return spans
