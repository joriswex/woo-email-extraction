"""
Plot per-dossier F1 (IoU >= 0.5) for all four Stage-1 approaches plus
their average as a grouped bar chart, sorted by mean F1 and with
anonymized dossier labels (Dossier 1, Dossier 2, ...).

See results.md, "Per-dossier difficulty (cross-approach)".

Also writes a CSV table with the same data for pasting into
Word/Excel.

Pure Python / SVG output — no plotting library dependencies.

Usage
-----
    python scripts/plot_dossier_variance_bars.py
"""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
RESULTS = _ROOT / "data" / "results"
OUT_SVG = RESULTS / "stage1_dossier_variance_bars.svg"
OUT_CSV = RESULTS / "stage1_dossier_variance_bars.csv"

APPROACHES = [
    ("Regex", "#1b9e77"),
    ("RobBERT (line + proximity filter)", "#d95f02"),
    ("GPT-5.5 (text)", "#7570b3"),
    ("GPT-5.5 (vision+text)", "#e7298a"),
]
APPROACH_LABELS = {
    "RobBERT (line + proximity filter)": "RobBERT (line, unfiltered)",
}
AVERAGE_COLOR = "#999999"


def main() -> None:
    rows = list(csv.DictReader(open(RESULTS / "stage1_per_dossier.csv")))
    by_dossier: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        by_dossier[r["dossier_id"]][r["approach"]] = float(r["f1"])

    dossiers = sorted(
        by_dossier.keys(),
        key=lambda d: -sum(by_dossier[d].get(a, 0.0) for a, _ in APPROACHES) / len(APPROACHES),
    )
    # Anonymized labels, in the same (difficulty-sorted) order
    anon_labels = {did: f"Dossier {i+1}" for i, did in enumerate(dossiers)}

    # --- CSV table for Word/Excel ---
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
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
    print(f"Written to {OUT_CSV.relative_to(_ROOT)}")

    # --- layout ---
    W, H = 800, 460
    margin_left, margin_right = 60, 200
    margin_top, margin_bottom = 30, 90
    plot_w = W - margin_left - margin_right
    plot_h = H - margin_top - margin_bottom

    n = len(dossiers)
    n_approaches = len(APPROACHES) + 1  # + average bar
    group_w = plot_w / n
    bar_pad = 4
    bar_w = (group_w - 2 * bar_pad) / n_approaches
    y_min, y_max = 0.0, 1.0

    def y_pos(f1: float) -> float:
        return margin_top + plot_h * (1 - (f1 - y_min) / (y_max - y_min))

    svg_parts: list[str] = []
    svg_parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif" font-size="12">'
    )
    svg_parts.append(f'<rect width="{W}" height="{H}" fill="white"/>')

    # Title
    svg_parts.append(
        f'<text x="{W/2}" y="18" text-anchor="middle" font-size="14" font-weight="bold">'
        f'Stage 1 F1 (IoU ≥ 0.5) per dossier, by approach</text>'
    )

    # Y gridlines + labels
    for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = y_pos(tick)
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left+plot_w}" y2="{y:.1f}" '
            f'stroke="#e0e0e0" stroke-width="1"/>'
        )
        svg_parts.append(
            f'<text x="{margin_left-8}" y="{y+4:.1f}" text-anchor="end" fill="#555">{tick:.1f}</text>'
        )

    # Axes
    svg_parts.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top+plot_h}" stroke="#333"/>'
    )
    svg_parts.append(
        f'<line x1="{margin_left}" y1="{margin_top+plot_h}" x2="{margin_left+plot_w}" y2="{margin_top+plot_h}" stroke="#333"/>'
    )

    # Bars + group separators + x labels
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

        # Average bar
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

    # Legend
    legend_x = margin_left + plot_w + 20
    legend_y = margin_top + 10
    for j, (approach, color) in enumerate(APPROACHES):
        ly = legend_y + j * 22
        svg_parts.append(f'<rect x="{legend_x}" y="{ly-8}" width="20" height="12" fill="{color}"/>')
        label = APPROACH_LABELS.get(approach, approach)
        svg_parts.append(f'<text x="{legend_x+28}" y="{ly+2}" font-size="11">{label}</text>')
    ly = legend_y + len(APPROACHES) * 22
    svg_parts.append(f'<rect x="{legend_x}" y="{ly-8}" width="20" height="12" fill="{AVERAGE_COLOR}"/>')
    svg_parts.append(f'<text x="{legend_x+28}" y="{ly+2}" font-size="11">Average (4 approaches)</text>')

    svg_parts.append(
        f'<text x="{margin_left-45}" y="{margin_top+plot_h/2}" text-anchor="middle" '
        f'transform="rotate(-90 {margin_left-45} {margin_top+plot_h/2})">F1</text>'
    )

    svg_parts.append("</svg>")

    OUT_SVG.write_text("\n".join(svg_parts), encoding="utf-8")
    print(f"Written to {OUT_SVG.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
