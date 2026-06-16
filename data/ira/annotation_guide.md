# Annotation Guide for Secondary Rater

This guide describes the two annotation tasks for the inter-rater agreement
study, carried out on two dossiers from a selection of Dutch government email's
released under the Wet open overheid (Woo). The government uses redaction codes
to protect personal information, these usually look like '5.1.2.a' or similar.
These are the field values and should therefore always be included. The text was
extracted using OCR, so some text may look garbled and unreadable. Please use
your best judgment in this case!

## Stage 1: email boundary detection

Each dossier consists of one long text in which multiple emails appear one after
another. The task is to mark the boundary of every individual email, drawing a
span from the first character of its header block to the last character of its
body.

An email's header block typically begins with a line such as Van:, From:, To:,
or Aan:, followed by Verzonden:/Sent:, Onderwerp:/ Subject:, and so on. Mark a
new email wherever such a header block begins and a clear transition to a new
message is visible.

Text within a reply (lines starting with > or indented quotes) and forwarded
headers that appear inside the body of another message should not be marked
separately — they belong to the parent email. The same applies to signature
blocks and disclaimers at the end of a message.

A few situations that may appear:

- If a header is partly unreadable by OCR, mark the email as well as you can,
  using whichever header field is still readable/interpretable as the anchor.
- If two emails run into each other without a clear separator, use your own
  judgement about where the new email begins.
- A forwarded message with its own header should only be marked separately if
  it appears at the top level of the dossier, not when it is nested inside
  another email.

## Stage 2: field extraction

For each email, highlight the value belonging to each of the following fields:
FROM, TO, CC, DATE, SUBJECT, and ATTACHMENT. Annotate only the value, not the
field label itself — for Van: Jan Jansen <jan@example.nl>, the annotated span
is Jan Jansen <jan@example.nl>, not the 'Van:' part.

| Label | Dutch header | English header | What to annotate |
|---|---|---|---|
| FROM | Van: | From: | Sender name and/or email address |
| TO | Aan: | To: | Recipient name(s) and/or email address(es) |
| CC | CC: / Cc: | CC: / Cc: | CC recipient(s), name(s) and/or address(es) |
| DATE | Verzonden: | Sent: / Date: | The date and time value |
| SUBJECT | Onderwerp: | Subject: | The subject line text |
| ATTACHMENT | Bijlage: / filename in body | Attachment: | The attachment filename(s) |

For FROM, TO, and CC, include the full value — both name and email address
where both are present. If several recipients are listed on one line, annotate
that line as a single span. Redacted names such as 5.1.2e are still annotated,
since the field itself is present even though its value has been hidden.

For SUBJECT, include the full subject text, including any RE: or FW: prefixes.

A few situations that come up sometimes:

- If a field is missing entirely, or its label is present but there is nothing
  in the value, leave it unannotated.
- If an email unusually has more than one FROM address, annotate each one
  separately.

If you're unsure about a particular case, use your best judgement. There are no
wrong answers!
