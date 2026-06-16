"""
Plot per-dossier F1 bar charts for Stage 1 and Stage 2.

Stage 1: plots F1 (IoU >= 0.5) for four approaches + average, reading from
         data/results/stage1_per_dossier.csv (produced by scripts/evaluate.py).

Stage 2: plots macro F1 (exact match or ANLS*) for five approaches + average,
         reading from data/results/stage2_raw_predictions.json and
         data/annotations/stage2_test.json.

Both stages output an SVG bar chart and a CSV summary table to data/results/.

Pure Python / SVG output — no plotting library dependencies.

Usage
-----
    python scripts/metrics/plot_dossier_variance.py --stage 1
    python scripts/metrics/plot_dossier_variance.py --stage 2
    python scripts/metrics/plot_dossier_variance.py --stage 2 --metric anls
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from eval_metrics import accumulate_field_counts, empty_counts, prf_from_counts

DATA_DIR = _ROOT / "data"
RESULTS = DATA_DIR / "results"

# Stage 1: key is the CSV label from evaluate.py; STAGE1_APPROACH_LABELS remaps it for display
STAGE1_APPROACHES = [
    ("Regex", "#1b9e77"),
    ("RobBERT (line + proximity filter)", "#d95f02"),
    ("GPT-5.5 (text)", "#7570b3"),
    ("GPT-5.5 (vision+text)", "#e7298a"),
]
STAGE1_APPROACH_LABELS = {
    "RobBERT (line + proximity filter)": "RobBERT (line, unfiltered)",
}

STAGE2_APPROACHES = [
    ("Regex", "#1b9e77"),
    ("BERT", "#66a61e"),
    ("BERT (weighted)", "#d95f02"),
    ("GPT-5.5 (text)", "#7570b3"),
    ("GPT-5.5 (vision+text)", "#e7298a"),
]
STAGE2_APPROACH_LABELS = {
    "BERT": "RobBERT (unweighted)",
    "BERT (weighted)": "RobBERT (class-weighted)",
}

EVAL_FIELDS = ["FROM", "TO", "CC", "DATE", "SUBJECT", "ATTACHMENT"]
AVERAGE_COLOR = "#999999"


def _gt_values(record: dict) -> list[dict]:
    text = record["text"]
    return [
        {"label": f["label"], "value": text[f["start_char"]:f["end_char"]].strip()}
        for f in record["fields"]
        if f["label"] in EVAL_FIELDS
    ]


def _render_svg(
    by_dossier: dict[str, dict[str, float]],
    approaches: list[tuple[str, str]],
    approach_labels: dict[str, str],
    title: str,
    y_label: str,
    out_svg: Path,
    out_csv: Path,
) -> None:
    dossiers = sorted(
        by_dossier.keys(),
        key=lambda d: -sum(by_dossier[d].get(a, 0.0) for a, _ in approaches) / len(approaches),
    )
    anon_labels = {did: f"Dossier {i+1}" for i, did in enumerate(dossiers)}

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Dossier"] + [approach_labels.get(a, a) for a, _ in approaches]
            + ["Average", "Range (max-min)", "Std dev"]
        )
        for did in dossiers:
            f1s = [by_dossier[did].get(a, 0.0) for a, _ in approaches]
            avg = sum(f1s) / len(f1s)
            rng = max(f1s) - min(f1s)
            std = statistics.pstdev(f1s)
            writer.writerow(
                [anon_labels[did]] + [f"{v:.3f}" for v in f1s]
                + [f"{avg:.3f}", f"{rng:.3f}", f"{std:.3f}"]
            )
    print(f"Written to {out_csv.relative_to(_ROOT)}")

    n_bars = len(approaches) + 1  # + average bar
    W = 900 if len(approaches) > 4 else 800
    H = 460
    margin_left, margin_right = 60, 200
    margin_top, margin_bottom = 30, 90
    plot_w = W - margin_left - margin_right
    plot_h = H - margin_top - margin_bottom

    group_w = plot_w / len(dossiers)
    bar_pad = 3 if len(approaches) > 4 else 4
    bar_w = (group_w - 2 * bar_pad) / n_bars
    y_min, y_max = 0.0, 1.0

    def y_pos(f1: float) -> float:
        return margin_top + plot_h * (1 - (f1 - y_min) / (y_max - y_min))

    svg: list[str] = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif" font-size="12">'
    )
    svg.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    svg.append(
        f'<text x="{W/2}" y="18" text-anchor="middle" font-size="14" font-weight="bold">'
        f'{title}</text>'
    )

    for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = y_pos(tick)
        svg.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left+plot_w}" y2="{y:.1f}" '
            f'stroke="#e0e0e0" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{margin_left-8}" y="{y+4:.1f}" text-anchor="end" fill="#555">{tick:.1f}</text>'
        )

    svg.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top+plot_h}" stroke="#333"/>'
    )
    svg.append(
        f'<line x1="{margin_left}" y1="{margin_top+plot_h}" x2="{margin_left+plot_w}" y2="{margin_top+plot_h}" stroke="#333"/>'
    )

    for i, did in enumerate(dossiers):
        group_x0 = margin_left + i * group_w
        group_center = group_x0 + group_w / 2

        if i > 0:
            svg.append(
                f'<line x1="{group_x0:.1f}" y1="{margin_top}" x2="{group_x0:.1f}" y2="{margin_top+plot_h}" '
                f'stroke="#f0f0f0" stroke-width="1"/>'
            )

        for j, (approach, color) in enumerate(approaches):
            v = by_dossier[did].get(approach, 0.0)
            x = group_x0 + bar_pad + j * bar_w
            y = y_pos(v)
            h = margin_top + plot_h - y
            svg.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-1:.1f}" height="{h:.1f}" fill="{color}"/>'
            )

        avg = sum(by_dossier[did].get(a, 0.0) for a, _ in approaches) / len(approaches)
        x = group_x0 + bar_pad + len(approaches) * bar_w
        y = y_pos(avg)
        h = margin_top + plot_h - y
        svg.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-1:.1f}" height="{h:.1f}" fill="{AVERAGE_COLOR}"/>'
        )

        svg.append(
            f'<text x="{group_center:.1f}" y="{margin_top+plot_h+18}" text-anchor="end" '
            f'transform="rotate(-40 {group_center:.1f} {margin_top+plot_h+18})">{anon_labels[did]}</text>'
        )

    legend_x = margin_left + plot_w + 20
    legend_y = margin_top + 10
    for j, (approach, color) in enumerate(approaches):
        ly = legend_y + j * 22
        svg.append(f'<rect x="{legend_x}" y="{ly-8}" width="20" height="12" fill="{color}"/>')
        label = approach_labels.get(approach, approach)
        svg.append(f'<text x="{legend_x+28}" y="{ly+2}" font-size="11">{label}</text>')
    ly = legend_y + len(approaches) * 22
    svg.append(f'<rect x="{legend_x}" y="{ly-8}" width="20" height="12" fill="{AVERAGE_COLOR}"/>')
    svg.append(f'<text x="{legend_x+28}" y="{ly+2}" font-size="11">Average ({len(approaches)} approaches)</text>')

    svg.append(
        f'<text x="{margin_left-45}" y="{margin_top+plot_h/2}" text-anchor="middle" '
        f'transform="rotate(-90 {margin_left-45} {margin_top+plot_h/2})">{y_label}</text>'
    )

    svg.append("</svg>")
    out_svg.write_text("\n".join(svg), encoding="utf-8")
    print(f"Written to {out_svg.relative_to(_ROOT)}")


def stage1() -> None:
    rows = list(csv.DictReader(open(RESULTS / "stage1_per_dossier.csv")))
    by_dossier: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        by_dossier[r["dossier_id"]][r["approach"]] = float(r["f1"])

    _render_svg(
        by_dossier=dict(by_dossier),
        approaches=STAGE1_APPROACHES,
        approach_labels=STAGE1_APPROACH_LABELS,
        title="Stage 1 F1 (IoU ≥ 0.5) per dossier, by approach",
        y_label="F1",
        out_svg=RESULTS / "stage1_dossier_variance_bars.svg",
        out_csv=RESULTS / "stage1_dossier_variance_bars.csv",
    )


def stage2(metric: str = "exact") -> None:
    score_key = "exact_f1" if metric == "exact" else "anls_f1"
    suffix = "" if metric == "exact" else "_anls"
    title_metric = "exact" if metric == "exact" else "ANLS*"
    y_label = "F1" if metric == "exact" else "ANLS*"

    test_data = json.load(open(DATA_DIR / "annotations" / "stage2_test.json", encoding="utf-8"))
    gt_by_email = {(r["dossier_id"], r["email_id"]): _gt_values(r) for r in test_data}
    raw = json.load(open(RESULTS / "stage2_raw_predictions.json", encoding="utf-8"))

    preds_by_key: dict[tuple, list] = defaultdict(list)
    for r in raw:
        preds_by_key[(r["dossier_id"], r["email_id"], r["approach"])] = r["predictions"]

    dossiers_all = sorted({r["dossier_id"] for r in test_data})
    by_dossier: dict[str, dict[str, float]] = defaultdict(dict)
    for did in dossiers_all:
        emails = [eid for (d, eid) in gt_by_email if d == did]
        for approach, _ in STAGE2_APPROACHES:
            counters = empty_counts()
            for eid in emails:
                gt = gt_by_email[(did, eid)]
                preds = preds_by_key.get((did, eid, approach), [])
                accumulate_field_counts(counters, preds, gt, EVAL_FIELDS)
            per_field = prf_from_counts(counters)
            present = {f: v for f, v in per_field.items() if counters[f]["n_gt"] > 0}
            macro = sum(v[score_key] for v in present.values()) / len(present)
            by_dossier[did][approach] = macro

    _render_svg(
        by_dossier=dict(by_dossier),
        approaches=STAGE2_APPROACHES,
        approach_labels=STAGE2_APPROACH_LABELS,
        title=f"Stage 2 macro {title_metric} per dossier, by approach",
        y_label=y_label,
        out_svg=RESULTS / f"stage2_dossier_variance_bars{suffix}.svg",
        out_csv=RESULTS / f"stage2_dossier_variance_bars{suffix}.csv",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-dossier F1 bar charts for Stage 1 or Stage 2."
    )
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1,
                        help="Which stage to plot (default: 1)")
    parser.add_argument("--metric", choices=["exact", "anls"], default="exact",
                        help="Stage 2 score metric: exact F1 or ANLS* (default: exact)")
    args = parser.parse_args()

    if args.stage == 1:
        stage1()
    else:
        stage2(args.metric)


if __name__ == "__main__":
    main()
