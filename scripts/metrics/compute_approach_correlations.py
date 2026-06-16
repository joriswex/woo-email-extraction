"""
Pairwise Pearson correlation between approaches' per-dossier scores.

For two approaches A and B, this correlates A's score across the n
dossiers with B's score across the same n dossiers. A high positive
correlation means: on dossiers where A scores high, B also tends to
score high (and vice versa) -- i.e., the dossier itself sets a shared
difficulty level that both approaches respond to, regardless of any
fixed gap between A and B.

Usage
-----
    python scripts/metrics/compute_approach_correlations.py <csv-file> [...]
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent

NON_APPROACH_COLS = {"Dossier", "Average", "Range (max-min)", "Std dev"}


def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    sx = sum((v - mx) ** 2 for v in x) ** 0.5
    sy = sum((v - my) ** 2 for v in y) ** 0.5
    return cov / (sx * sy)


def correlations(path: Path) -> None:
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        approach_cols = [c for c in reader.fieldnames if c not in NON_APPROACH_COLS]
        rows = list(reader)

    series = {c: [float(row[c]) for row in rows] for c in approach_cols}

    print(f"=== {path.relative_to(_ROOT)} ({len(rows)} dossiers x {len(approach_cols)} approaches) ===")
    header = " " * 28 + "".join(f"{c[:10]:>11}" for c in approach_cols)
    print(header)
    for a in approach_cols:
        row = f"{a:<28}"
        for b in approach_cols:
            r = pearson(series[a], series[b])
            row += f"{r:11.2f}"
        print(row)
    print()

    # Average pairwise correlation (off-diagonal)
    pairs = []
    for i, a in enumerate(approach_cols):
        for b in approach_cols[i + 1:]:
            pairs.append(pearson(series[a], series[b]))
    avg_r = sum(pairs) / len(pairs)
    print(f"Average pairwise correlation: {avg_r:.3f}\n")


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        paths = [
            _ROOT / "data" / "results" / "stage1_dossier_variance_bars.csv",
            _ROOT / "data" / "results" / "stage2_dossier_variance_bars.csv",
            _ROOT / "data" / "results" / "stage2_dossier_variance_bars_anls.csv",
        ]
    for path in paths:
        correlations(path)


if __name__ == "__main__":
    main()
