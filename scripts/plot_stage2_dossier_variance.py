"""
Plot per-dossier macro F1 (exact match or ANLS*) for the four Stage-2
approaches plus their average as a grouped bar chart, sorted by average
score and with anonymized dossier labels (Dossier 1, Dossier 2, ...).

For each dossier, the macro score is the average of the per-field score
(exact-F1 or ANLS*) across the EVAL_FIELDS that actually occur (n_gt > 0)
in that dossier's ground truth -- fields absent from a dossier are
excluded rather than scored as zero.

Mirrors scripts/plot_dossier_variance_bars.py (Stage 1). Also writes a
CSV table with the same data for pasting into Word/Excel.

Pure Python / SVG output -- no plotting library dependencies.

Usage
-----
    python scripts/plot_stage2_dossier_variance.py              # exact F1
    python scripts/plot_stage2_dossier_variance.py --metric anls # ANLS*
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from eval_metrics import accumulate_field_counts, empty_counts, prf_from_counts

DATA_DIR = _ROOT / "data"
RESULTS = DATA_DIR / "results"
STAGE2_TEST_ANN = DATA_DIR / "annotations" / "stage2_test.json"
RAW_PREDS = RESULTS / "stage2_raw_predictions.json"

EVAL_FIELDS = ["FROM", "TO", "CC", "DATE", "SUBJECT", "ATTACHMENT"]

APPROACHES = [
    ("Regex", "#1b9e77"),
    ("BERT", "#66a61e"),
    ("BERT (weighted)", "#d95f02"),
    ("GPT-5.5 (text)", "#7570b3"),
    ("GPT-5.5 (vision+text)", "#e7298a"),
]
APPROACH_LABELS = {
    "BERT": "RobBERT (unweighted)",
    "BERT (weighted)": "RobBERT (class-weighted)",
}
AVERAGE_COLOR = "#999999"


def _gt_values(record: dict) -> list[dict]:
    text = record["text"]
    return [
        {"label": f["label"], "value": text[f["start_char"]:f["end_char"]].strip()}
        for f in record["fields"]
        if f["label"] in EVAL_FIELDS
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=["exact", "anls"], default="exact")
    args = parser.parse_args()
    score_key = "exact_f1" if args.metric == "exact" else "anls_f1"
    suffix = "" if args.metric == "exact" else "_anls"
    title_metric = "exact" if args.metric == "exact" else "ANLS*"
    out_svg = RESULTS / f"stage2_dossier_variance_bars{suffix}.svg"
    out_csv = RESULTS / f"stage2_dossier_variance_bars{suffix}.csv"

    test_data = json.load(open(STAGE2_TEST_ANN, encoding="utf-8"))
    gt_by_email: dict[tuple[str, str], list[dict]] = {
        (r["dossier_id"], r["email_id"]): _gt_values(r) for r in test_data
    }

    raw = json.load(open(RAW_PREDS, encoding="utf-8"))
    preds_by_key: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in raw:
        preds_by_key[(r["dossier_id"], r["email_id"], r["approach"])] = r["predictions"]

    dossiers_all = sorted({r["dossier_id"] for r in test_data})

    by_dossier: dict[str, dict[str, float]] = defaultdict(dict)
    for did in dossiers_all:
        emails = [eid for (d, eid) in gt_by_email if d == did]
        for approach, _ in APPROACHES:
            counters = empty_counts()
            for eid in emails:
                gt = gt_by_email[(did, eid)]
                preds = preds_by_key.get((did, eid, approach), [])
                accumulate_field_counts(counters, preds, gt, EVAL_FIELDS)
            per_field = prf_from_counts(counters)
            # Exclude fields absent from this dossier's ground truth
            present = {f: v for f, v in per_field.items() if counters[f]["n_gt"] > 0}
            macro = sum(v[score_key] for v in present.values()) / len(present)
            by_dossier[did][approach] = macro

    dossiers = sorted(
        by_dossier.keys(),
        key=lambda d: -sum(by_dossier[d].get(a, 0.0) for a, _ in APPROACHES) / len(APPROACHES),
    )
    anon_labels = {did: f"Dossier {i+1}" for i, did in enumerate(dossiers)}

    # --- CSV table for Word/Excel ---
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Dossier"] + [APPROACH_LABELS.get(a, a) for a, _ in APPROACHES]
            + ["Average", "Range (max-min)", "Std dev"]
        )
        for did in dossiers:
            f1s = [by_dossier[did].get(a, 0.0) for a, _ in APPROACHES]
            avg = sum(f1s) / len(f1s)
            rng = max(f1s) - min(f1s)
            std = statistics.pstdev(f1s)
            writer.writerow(
                [anon_labels[did]] + [f"{v:.3f}" for v in f1s]
                + [f"{avg:.3f}", f"{rng:.3f}", f"{std:.3f}"]
            )
    print(f"Written to {out_csv.relative_to(_ROOT)}")

    # --- layout ---
    W, H = 900, 460
    margin_left, margin_right = 60, 200
    margin_top, margin_bottom = 30, 90
    plot_w = W - margin_left - margin_right
    plot_h = H - margin_top - margin_bottom

    n = len(dossiers)
    n_bars = len(APPROACHES) + 1  # + average bar
    group_w = plot_w / n
    bar_pad = 3
    bar_w = (group_w - 2 * bar_pad) / n_bars
    y_min, y_max = 0.0, 1.0

    def y_pos(f1: float) -> float:
        return margin_top + plot_h * (1 - (f1 - y_min) / (y_max - y_min))

    svg_parts: list[str] = []
    svg_parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif" font-size="12">'
    )
    svg_parts.append(f'<rect width="{W}" height="{H}" fill="white"/>')

    svg_parts.append(
        f'<text x="{W/2}" y="18" text-anchor="middle" font-size="14" font-weight="bold">'
        f'Stage 2 macro {title_metric} per dossier, by approach</text>'
    )

    for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = y_pos(tick)
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left+plot_w}" y2="{y:.1f}" '
            f'stroke="#e0e0e0" stroke-width="1"/>'
        )
        svg_parts.append(
            f'<text x="{margin_left-8}" y="{y+4:.1f}" text-anchor="end" fill="#555">{tick:.1f}</text>'
        )

    svg_parts.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top+plot_h}" stroke="#333"/>'
    )
    svg_parts.append(
        f'<line x1="{margin_left}" y1="{margin_top+plot_h}" x2="{margin_left+plot_w}" y2="{margin_top+plot_h}" stroke="#333"/>'
    )

    for i, did in enumerate(dossiers):
        group_x0 = margin_left + i * group_w
        group_center = group_x0 + group_w / 2

        if i > 0:
            svg_parts.append(
                f'<line x1="{group_x0:.1f}" y1="{margin_top}" x2="{group_x0:.1f}" y2="{margin_top+plot_h}" '
                f'stroke="#f0f0f0" stroke-width="1"/>'
            )

        for j, (approach, color) in enumerate(APPROACHES):
            v = by_dossier[did].get(approach, 0.0)
            x = group_x0 + bar_pad + j * bar_w
            y = y_pos(v)
            h = margin_top + plot_h - y
            svg_parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-1:.1f}" height="{h:.1f}" fill="{color}"/>'
            )

        avg = sum(by_dossier[did].get(a, 0.0) for a, _ in APPROACHES) / len(APPROACHES)
        x = group_x0 + bar_pad + len(APPROACHES) * bar_w
        y = y_pos(avg)
        h = margin_top + plot_h - y
        svg_parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-1:.1f}" height="{h:.1f}" fill="{AVERAGE_COLOR}"/>'
        )

        svg_parts.append(
            f'<text x="{group_center:.1f}" y="{margin_top+plot_h+18}" text-anchor="end" '
            f'transform="rotate(-40 {group_center:.1f} {margin_top+plot_h+18})">{anon_labels[did]}</text>'
        )

    legend_x = margin_left + plot_w + 20
    legend_y = margin_top + 10
    for j, (approach, color) in enumerate(APPROACHES):
        ly = legend_y + j * 22
        svg_parts.append(f'<rect x="{legend_x}" y="{ly-8}" width="20" height="12" fill="{color}"/>')
        label = APPROACH_LABELS.get(approach, approach)
        svg_parts.append(f'<text x="{legend_x+28}" y="{ly+2}" font-size="11">{label}</text>')
    ly = legend_y + len(APPROACHES) * 22
    svg_parts.append(f'<rect x="{legend_x}" y="{ly-8}" width="20" height="12" fill="{AVERAGE_COLOR}"/>')
    svg_parts.append(f'<text x="{legend_x+28}" y="{ly+2}" font-size="11">Average ({len(APPROACHES)} approaches)</text>')

    y_label = "F1" if args.metric == "exact" else "ANLS*"
    svg_parts.append(
        f'<text x="{margin_left-45}" y="{margin_top+plot_h/2}" text-anchor="middle" '
        f'transform="rotate(-90 {margin_left-45} {margin_top+plot_h/2})">{y_label}</text>'
    )

    svg_parts.append("</svg>")

    out_svg.write_text("\n".join(svg_parts), encoding="utf-8")
    print(f"Written to {out_svg.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
