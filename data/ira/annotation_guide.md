# Annotation Guide — Inter-Rater Agreement Study

This guide covers annotation tasks for two dossiers from a corpus of Dutch
government email correspondence released under the Wet open overheid (Woo).
The text has been extracted via OCR and may contain extraction artefacts.
Redacted personal data appears as codes such as `5.1.2e`, `5.1.2.b`, `8.4.2e`.

---

## Stage 1 — Email Boundary Detection

**Task:** Mark the start of each individual email in the dossier text.

Each dossier is a single long text containing multiple emails concatenated
together. Your job is to draw a span from the very first character of each
email's header to the last character of that email's body.

### What counts as an email boundary?

An email starts at the **first character of its header block** — typically
the line beginning with `Van:`, `From:`, `To:`, `Aan:`, or `Verzonden:`.

**Mark a new email when you see:**
- A new header block with sender/recipient/date fields
- A clear transition from one message to the next

**Do NOT mark as a new email:**
- Inline quoted text within a reply (the `>` lines or indented quotes)
- Forwarded message headers embedded *inside* a message body
  (these are part of the parent email, not a separate one)
- Signature blocks or disclaimers at the end of a message

### Edge cases

| Situation | What to do |
|---|---|
| Header is partially garbled by OCR | Still mark it — use the first readable header field |
| Two emails appear without a clear separator | Use your best judgment; mark where you believe the new email starts |
| A forwarded message has its own header | Only mark it as a new email if it appears at the top level, not nested inside another email's body |

---

## Stage 2 — Field Extraction

**Task:** For each email, highlight the **value** of each of the following fields.

> **Important:** Annotate the VALUE only — do not include the field label itself.
> For example, for `Van: Jan Jansen <jan@example.nl>` annotate only `Jan Jansen <jan@example.nl>`.

### Field definitions

| Label | Dutch header | English header | What to annotate |
|---|---|---|---|
| `FROM` | Van: | From: | The sender name and/or email address |
| `TO` | Aan: | To: | The recipient name(s) and/or email address(es) |
| `CC` | CC: / Cc: | CC: / Cc: | The CC recipient(s) name(s) and/or address(es) |
| `DATE` | Verzonden: | Sent: / Date: | The date and time value |
| `SUBJECT` | Onderwerp: | Subject: | The subject line text |
| `ATTACHMENT` | Bijlage: / (filename in body) | Attachment: | The attachment filename(s) |

### Annotation rules

**FROM / TO / CC**
- Include the full value: name + email address if both are present
  (e.g., `Jan Jansen <jan@example.nl>`)
- If multiple recipients are on one line, annotate the entire value as one span
- If recipients are listed across multiple lines, annotate each line separately
- Redacted names (`5.1.2e`) are still annotated — the field is present even if the value is hidden

**DATE**
- Annotate only the date/time value, not the field label
- Include the full date string as it appears (e.g., `dinsdag 8 april 2025 14:57`)

**SUBJECT**
- Annotate the full subject text including any `RE:`, `FW:`, `RV:` prefixes

**ATTACHMENT**
- Annotate the filename only (e.g., `rapport_2024.pdf`)
- If no attachment is explicitly mentioned, leave this field unannotated
- Do not annotate vague references like "zie bijlage" (see attachment)

### Edge cases

| Situation | What to do |
|---|---|
| Field is missing entirely | Leave it unannotated |
| Field label is present but value is blank | Leave it unannotated |
| Multiple FROM addresses (rare) | Annotate each address separately |
| OCR has merged field label and value | Annotate as much of the value as you can identify |
| Same field appears twice in one email | Annotate both occurrences |

---

## General notes

- If you are unsure, **make your best judgment and continue** — note any
  systematic uncertainties so they can be discussed after the exercise.
- Do not consult the other annotator's work until both annotation sets are complete.
- Work through the dossiers in order; do not skip emails.
- Aim for consistency within your own annotations: if you make a decision
  on one email, apply the same rule to all similar emails.
