"""Render the manuscript comparison from the fitted pointwise effect curves."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


OUTPUT_DIR = Path("outputs/aggregation/summary")
INPUT_CSV = OUTPUT_DIR / "aggregation_effect_curves_pointwise.csv"
OUTPUT_STEM = "figure_2_aggregation"
PNG_DPI = 300
OUTPUT_FORMATS = ("pdf", "png")

EFFECTS = ("ME", "DE", "TE")
METHODS = ("truth", "latent", "oracle", "scalar")
TITLES = ("(a) Mediated effect", "(b) Direct effect", "(c) Total effect")
LINE_STYLES = {
    "truth": ("Truth", "black", "-", 2.65, 3),
    "latent": ("BLaFM", "black", "--", 2.25, 4),
    "oracle": ("Oracle", "0.48", "-.", 2.25, 4),
    "scalar": ("Donor-average scalar", "0.62", (0, (2.2, 1.8)), 2.25, 2),
}
PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "mathtext.fontset": "dejavusans",
    "text.usetex": False,
    "axes.titlesize": 13,
    "axes.titleweight": "normal",
    "axes.labelsize": 12.5,
    "axes.linewidth": 1.25,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "xtick.major.width": 1.25,
    "ytick.major.width": 1.25,
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}


def load_curves(path: Path) -> dict:
    """Read the saved pointwise summaries; do not recompute any effects."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"CSV not found: {path}\nSet INPUT_CSV or pass --input-csv. "
            "Use aggregation_effect_curves_pointwise.csv from the paper analysis."
        )
    columns = ["pseudotime"] + [f"{m}_{e}" for e in EFFECTS for m in METHODS]
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(columns) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            try:
                values = tuple(float(row[k]) for k in columns)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Row {row_number}: invalid curve values") from exc
            if not np.isfinite(values).all():
                raise ValueError(f"Row {row_number}: curve values must be finite")
            rows.append(values)
    if len(rows) < 2:
        raise ValueError("The CSV must contain at least two grid points")
    values = np.asarray(sorted(rows, key=lambda r: r[0]), dtype=float)
    grid = values[:, 0]
    if np.any(np.diff(grid) <= 0):
        raise ValueError("Duplicate grid points in pseudotime")
    if not np.allclose(grid[[0, -1]], [0, 1], atol=1e-10, rtol=0):
        raise ValueError("The pseudotime grid must span [0, 1]")
    curves = {"grid": grid}
    for effect in EFFECTS:
        curves[effect] = {m: values[:, columns.index(f"{m}_{effect}")] for m in METHODS}
        scalar = curves[effect]["scalar"]
        if not np.allclose(scalar, scalar[0], atol=1e-10, rtol=0):
            raise ValueError(f"scalar_{effect} must contain a constant horizontal reference")
    return curves


def plot_curves(
    curves: dict, output_dir: Path, dpi: int = PNG_DPI, formats: tuple = OUTPUT_FORMATS
) -> list[Path]:
    """Draw the saved curves, horizontal scalar summaries, and zero references."""
    if dpi <= 0 or not formats or any(f not in {"pdf", "png", "svg"} for f in formats):
        raise ValueError("Use a positive dpi and formats chosen from pdf, png, svg")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.1))
        try:
            grid = curves["grid"]
            for ax, effect, title in zip(axes, EFFECTS, TITLES):
                ax.axhline(0, color="0.80", linewidth=0.75, linestyle=(0, (2, 2)), zorder=1)
                for method in ("scalar", "truth", "oracle", "latent"):
                    label, color, linestyle, width, zorder = LINE_STYLES[method]
                    ax.plot(
                        grid,
                        curves[effect][method],
                        label=label,
                        color=color,
                        linestyle=linestyle,
                        linewidth=width,
                        zorder=zorder,
                    )
                ax.set_title(title, loc="left", pad=8)
                ax.set_xlabel(r"State $t$", labelpad=4)
                ax.set_ylabel(r"$\mathrm{" + effect + r"}(t)$", labelpad=4)
                ax.set_xlim(0, 1)
                ax.set_xticks(np.linspace(0, 1, 6))
                ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                ax.yaxis.set_major_locator(MultipleLocator(0.1))
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction="out", pad=4)
                ax.grid(False)
                all_y = np.concatenate(([0.0], *[curves[effect][m] for m in METHODS]))
                span = float(np.ptp(all_y))
                padding = span * 0.08 if span > 0 else 0.08
                ax.set_ylim(float(all_y.min()) - padding, float(all_y.max()) + padding)

            handles, labels = axes[0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            order = ["Truth", "BLaFM", "Oracle", "Donor-average scalar"]
            legend = fig.legend(
                [by_label[label] for label in order],
                order,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.015),
                ncol=4,
                frameon=False,
                handlelength=3.2,
                handletextpad=0.75,
                columnspacing=1.3,
                borderaxespad=0,
            )
            for line in legend.get_lines():
                line.set_linewidth(2.7 if line.get_label() == "Truth" else 2.3)
            fig.subplots_adjust(left=0.066, right=0.96, bottom=0.225, top=0.905, wspace=0.31)
            for extension in dict.fromkeys(formats):
                path = output_dir / f"{OUTPUT_STEM}.{extension}"
                fig.savefig(path, dpi=dpi, facecolor="white")
                saved.append(path)
        finally:
            plt.close(fig)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=PNG_DPI)
    parser.add_argument(
        "--formats", nargs="+", choices=("pdf", "png", "svg"), default=OUTPUT_FORMATS
    )
    args = parser.parse_args()
    curves = load_curves(args.input_csv)
    for path in plot_curves(curves, args.output_dir, args.dpi, tuple(args.formats)):
        print(f"Saved: {path.resolve()}")


if __name__ == "__main__":
    main()
