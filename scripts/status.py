"""
Dossier annotation status overview.

Shows which dossiers have been pre-annotated, uploaded to Label Studio,
and fully annotated (stage 1 / stage 2).

Usage
-----
    python scripts/status.py
"""

from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

_ROOT     = Path(__file__).parent.parent
_RAW      = _ROOT / "data" / "raw"
_IMPORT   = _ROOT / "data" / "import"
_ANN      = _ROOT / "data" / "annotations"


def _load_json(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_metadata() -> dict[str, dict]:
    meta = {}
    for p in _RAW.glob("*.json"):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        meta[d["government_id"]] = d
    return meta


def main() -> None:
    meta     = _load_metadata()
    s1_ann   = {r["dossier_id"]: r for r in _load_json(_ANN / "stage1.json")}
    s2_ann   = defaultdict(int)
    for r in _load_json(_ANN / "stage2_train.json") + _load_json(_ANN / "stage2_test.json"):
        s2_ann[r["dossier_id"]] += 1

    # Collect all known IDs from every source
    all_ids = (
        set(meta.keys())
        | {p.stem for p in _RAW.glob("*.pdf")}
        | {p.stem for p in _IMPORT.glob("*.json") if _IMPORT.exists()}
        | set(s1_ann.keys())
        | set(s2_ann.keys())
    )

    col_w = [12, 7, 46, 12, 8, 12, 12]
    headers = ["ID", "Min.", "Title", "Published", "PDF", "Pre-annot.", "Stage1 (emails)", "Stage2 (emails)"]

    def row_sep():
        print("+" + "+".join("-" * (w + 2) for w in [12, 7, 46, 12, 8, 12, 16, 16]) + "+")

    def fmt(val, w):
        s = str(val)
        return s[:w].ljust(w)

    row_sep()
    print("| " + " | ".join(
        fmt(h, w) for h, w in zip(headers, [12, 7, 46, 12, 8, 12, 16, 16])
    ) + " |")
    row_sep()

    for gid in sorted(all_ids):
        m          = meta.get(gid, {})
        title      = m.get("title", "—")
        ministry   = m.get("ministry", "—")
        pub_date   = m.get("publish_date", "—")
        has_pdf    = "yes" if (_RAW / f"{gid}.pdf").exists() else "NO"
        has_import = "yes" if _IMPORT.exists() and (_IMPORT / f"{gid}.json").exists() else "—"

        if gid in s1_ann:
            n_emails = len(s1_ann[gid]["emails"])
            s1_status = f"done ({n_emails})"
        else:
            s1_status = "—"

        s2_count  = s2_ann.get(gid, 0)
        s2_status = f"done ({s2_count})" if s2_count else "—"

        print("| " + " | ".join(fmt(v, w) for v, w in zip(
            [gid, ministry, title, pub_date, has_pdf, has_import, s1_status, s2_status],
            [12, 7, 46, 12, 8, 12, 16, 16],
        )) + " |")

    row_sep()

    # Summary
    total     = len(all_ids)
    annotated = len(s1_ann)
    pending   = total - annotated
    print(f"\n  Total dossiers : {total}")
    print(f"  Stage-1 done   : {annotated}  ({', '.join(sorted(s1_ann))})")
    print(f"  Awaiting annot.: {pending}")
    if pending:
        waiting = sorted(all_ids - set(s1_ann))
        for gid in waiting:
            m = meta.get(gid, {})
            imp = "pre-annot. ready" if _IMPORT.exists() and (_IMPORT / f"{gid}.json").exists() else "not pre-annotated"
            print(f"    {gid}  [{m.get('ministry','?')}]  {m.get('title','')[:50]}  →  {imp}")


if __name__ == "__main__":
    main()
