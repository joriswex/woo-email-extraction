"""
Convert Label Studio export JSON to internal annotation schemas.

Label Studio exports a JSON array where each element is one annotated task:
    {
      "id": 123,
      "data": {"text": "...", ...},
      "annotations": [
        {
          "result": [
            {
              "type": "labels",
              "value": {"start": 5, "end": 18, "text": "...", "labels": ["FROM"]}
            }
          ]
        }
      ]
    }

Three modes
-----------
--stage 1  (dossier-level boundary annotations)
    Each task is one dossier.  Result items of type "labels" with label "EMAIL"
    become email boundary spans.  Requires "dossier_id" in task data.

    Output (our internal stage-1 schema):
        [{"dossier_id": ..., "text": ...,
          "emails": [{"email_id": "email_0", "start_char": ..., "end_char": ...}]}]

--stage 2  (email-level field annotations, separate two-pass workflow)
    Each task is one email.  Result items of type "labels" become field spans;
    result item of type "choices" becomes the email_type.
    Requires "dossier_id" and "email_id" in task data.

    Output (our internal stage-2 schema):
        [{"dossier_id": ..., "email_id": ..., "text": ...,
          "email_type": "NEW"|"REPLY"|"FORWARD",
          "fields": [{"label": ..., "start_char": ..., "end_char": ...}]}]

--stage combined  (single-pass annotation using labeling_config_combined.xml)
    Each task is one dossier.  Annotators mark EMAIL boundary spans AND field
    spans (FROM, TO, DATE, etc.) in one pass.  Email type is inferred
    automatically from the SUBJECT span text.

    Produces TWO output files:
        --output       stage-1 JSON (email boundary spans)
        --output_stage2  stage-2 JSON (field spans per email, email-relative offsets)

Usage
-----
    python scripts/labelstudio/convert_labelstudio_export.py \\
        --stage 1 \\
        --input  exports/stage1_export.json \\
        --output data/annotations/stage1.json

    python scripts/labelstudio/convert_labelstudio_export.py \\
        --stage 2 \\
        --input  exports/stage2_export.json \\
        --output data/annotations/stage2.json

    python scripts/labelstudio/convert_labelstudio_export.py \\
        --stage combined \\
        --input  exports/combined_export.json \\
        --output data/annotations/stage1.json \\
        --output_stage2 data/annotations/stage2.json
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SUBJECT_RE = re.compile(r"^Onderwerp:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
_REPLY_RE   = re.compile(r"^Re\s*:",   re.IGNORECASE)
_FORWARD_RE = re.compile(r"^(Fw\s*:|Fwd\s*:|Doorsturen\s*:)", re.IGNORECASE)


def _classify_email_type(text: str) -> str:
    """Rule-based email type from subject line (mirrors data_field.classify_email_type)."""
    m = _SUBJECT_RE.search(text)
    if m is None:
        return "NEW"
    subject = m.group(1).strip()
    if _REPLY_RE.match(subject):
        return "REPLY"
    if _FORWARD_RE.match(subject):
        return "FORWARD"
    return "NEW"


def _classify_from_subject_value(subject_value: str) -> str:
    """Classify email type directly from an already-extracted subject string."""
    s = subject_value.strip()
    if _REPLY_RE.match(s):
        return "REPLY"
    if _FORWARD_RE.match(s):
        return "FORWARD"
    return "NEW"


def _collect_spans(task: dict) -> list[tuple[int, int, str]]:
    """Return all (start, end, label) spans from a task's annotations."""
    spans = []
    for annotation in task.get("annotations", []):
        for result in annotation.get("result", []):
            if result.get("type") != "labels":
                continue
            value = result["value"]
            labels = value.get("labels", [])
            if labels:
                spans.append((value["start"], value["end"], labels[0]))
    return spans


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------


def _convert_stage1_task(task: dict) -> dict:
    data = task["data"]
    dossier_id = data["dossier_id"]
    text = data["text"]

    raw_spans = [
        (s, e) for s, e, label in _collect_spans(task) if label == "EMAIL"
    ]
    raw_spans.sort(key=lambda x: x[0])
    emails = [
        {"email_id": f"email_{i}", "start_char": s, "end_char": e}
        for i, (s, e) in enumerate(raw_spans)
    ]
    return {"dossier_id": dossier_id, "text": text, "emails": emails}


def convert_stage1(tasks: list[dict]) -> list[dict]:
    return [_convert_stage1_task(t) for t in tasks]


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------


def _convert_stage2_task(task: dict) -> dict:
    data = task["data"]
    dossier_id = data["dossier_id"]
    email_id = data["email_id"]
    text = data["text"]

    fields: list[dict] = []
    email_type = "NEW"

    for annotation in task.get("annotations", []):
        for result in annotation.get("result", []):
            result_type = result.get("type")
            value = result["value"]
            if result_type == "labels":
                labels = value.get("labels", [])
                if labels:
                    fields.append(
                        {"label": labels[0], "start_char": value["start"], "end_char": value["end"]}
                    )
            elif result_type == "choices":
                choices = value.get("choices", [])
                if choices:
                    email_type = choices[0]

    fields.sort(key=lambda f: f["start_char"])
    return {
        "dossier_id": dossier_id,
        "email_id": email_id,
        "text": text,
        "email_type": email_type,
        "fields": fields,
    }


def convert_stage2(tasks: list[dict]) -> list[dict]:
    return [_convert_stage2_task(t) for t in tasks]


# ---------------------------------------------------------------------------
# Combined (single-pass)
# ---------------------------------------------------------------------------


def _convert_combined_task(task: dict) -> tuple[dict, list[dict]]:
    """
    Convert one combined-annotation task into a stage-1 dict and a list of
    stage-2 dicts (one per email).

    Field span offsets in the export are relative to the dossier text.
    This function converts them to be relative to each email's own text slice,
    as required by the stage-2 schema.
    """
    data = task["data"]
    dossier_id = data["dossier_id"]
    text = data["text"]

    all_spans = _collect_spans(task)

    # Split into EMAIL boundary spans and field spans
    email_spans = sorted(
        [(s, e) for s, e, label in all_spans if label == "EMAIL"],
        key=lambda x: x[0],
    )
    field_spans = [(s, e, label) for s, e, label in all_spans if label != "EMAIL"]

    # Stage-1 output
    emails = [
        {"email_id": f"email_{i}", "start_char": s, "end_char": e}
        for i, (s, e) in enumerate(email_spans)
    ]
    stage1 = {"dossier_id": dossier_id, "text": text, "emails": emails}

    # Stage-2 output: one record per email
    stage2_list: list[dict] = []
    for email_dict in emails:
        email_id  = email_dict["email_id"]
        e_start   = email_dict["start_char"]
        e_end     = email_dict["end_char"]
        email_text = text[e_start:e_end]

        # Keep field spans that fall within this email; make offsets email-relative
        email_fields = []
        for f_start, f_end, f_label in field_spans:
            if f_start >= e_start and f_end <= e_end:
                email_fields.append(
                    {
                        "label": f_label,
                        "start_char": f_start - e_start,
                        "end_char":   f_end   - e_start,
                    }
                )
        email_fields.sort(key=lambda f: f["start_char"])

        # Infer email type: prefer the annotated SUBJECT value, fall back to
        # scanning the full email text
        subject_fields = [f for f in email_fields if f["label"] == "SUBJECT"]
        if subject_fields:
            subject_value = email_text[
                subject_fields[0]["start_char"]: subject_fields[0]["end_char"]
            ]
            email_type = _classify_from_subject_value(subject_value)
        else:
            email_type = _classify_email_type(email_text)

        stage2_list.append(
            {
                "dossier_id": dossier_id,
                "email_id":   email_id,
                "text":       email_text,
                "email_type": email_type,
                "fields":     email_fields,
            }
        )

    return stage1, stage2_list


def convert_combined(tasks: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Convert combined-annotation tasks.

    Returns
    -------
    stage1 : list of dossier-level annotation dicts
    stage2 : flat list of email-level annotation dicts
    """
    stage1_out: list[dict] = []
    stage2_out: list[dict] = []
    for task in tasks:
        s1, s2_list = _convert_combined_task(task)
        stage1_out.append(s1)
        stage2_out.extend(s2_list)
    return stage1_out, stage2_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert Label Studio export JSON to internal annotation schemas."
    )
    parser.add_argument(
        "--stage",
        choices=["1", "2", "combined"],
        required=True,
        help="1: boundary annotations, 2: field annotations, combined: both in one pass.",
    )
    parser.add_argument("--input",  required=True, help="Label Studio export JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file (stage-1 for combined)")
    parser.add_argument(
        "--output_stage2",
        default=None,
        help="[combined only] Output JSON file for stage-2 annotations",
    )
    args = parser.parse_args(argv)

    if args.stage == "combined" and not args.output_stage2:
        parser.error("--output_stage2 is required when --stage combined")

    with open(args.input, encoding="utf-8") as f:
        tasks: list[dict] = json.load(f)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.stage == "1":
        converted = convert_stage1(tasks)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)
        print(f"Converted {len(tasks)} tasks (stage 1) → {args.output}")

    elif args.stage == "2":
        converted = convert_stage2(tasks)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)
        print(f"Converted {len(tasks)} tasks (stage 2) → {args.output}")

    else:  # combined
        stage1, stage2 = convert_combined(tasks)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(stage1, f, ensure_ascii=False, indent=2)
        Path(args.output_stage2).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_stage2, "w", encoding="utf-8") as f:
            json.dump(stage2, f, ensure_ascii=False, indent=2)
        total_emails = sum(len(d["emails"]) for d in stage1)
        print(f"Converted {len(tasks)} tasks (combined)")
        print(f"  Stage-1 ({len(stage1)} dossiers, {total_emails} emails) → {args.output}")
        print(f"  Stage-2 ({len(stage2)} email records)                   → {args.output_stage2}")


if __name__ == "__main__":
    main()
