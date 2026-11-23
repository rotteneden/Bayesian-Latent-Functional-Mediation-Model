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


OUTPUT_DIR = Path("outputs/two_stage/summary")
INPUT_CSV = OUTPUT_DIR / "effect_curve_summary.csv"
OUTPUT_STEM = "figure_3_two_stage"
PNG_DPI = 300
OUTPUT_FORMATS = ("pdf", "png")

EFFECTS = ("ME", "DE", "TE")
METHODS = ("joint", "twostage_t", "twostage_coef", "oracle_t")
TITLES = ("(a) Mediated effect", "(b) Direct effect", "(c) Total effect")
LINE_STYLES = {
    "joint": ("BLaFM", "black", "--", 2.25),
    "twostage_t": ("Two-stage (states)", "0.50", ":", 2.25),
    "twostage_coef": ("Two-stage (coefficients)", "0.25", (0, (6, 2, 1.2, 2, 1.2, 2)), 2.20),
    "oracle_t": ("Oracle", "0.48", "-.", 2.25),
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
    """Read and validate all four methods; preserve their saved curve values."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"CSV not found: {path}\nSet INPUT_CSV or pass --input-csv. "
            "Use summary/effect_curve_summary.csv from the paper analysis."
        )
    groups = {(m, e): [] for m in METHODS for e in EFFECTS}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"method", "effect", "t", "truth", "mean_estimate"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            key = ((row.get("method") or "").strip(), (row.get("effect") or "").strip())
            if key not in groups:
                raise ValueError(f"Row {row_number}: unexpected method/effect {key}")
            try:
                values = tuple(float(row[k]) for k in ("t", "truth", "mean_estimate"))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Row {row_number}: invalid curve values") from exc
            if not np.isfinite(values).all():
                raise ValueError(f"Row {row_number}: curve values must be finite")
            groups[key].append(values)

    curves = {}
    for effect in EFFECTS:
        curves[effect] = {}
        reference_grid = reference_truth = None
        for method in METHODS:
            rows = groups[(method, effect)]
            if len(rows) < 2:
                raise ValueError(f"Missing or insufficient rows for {method}/{effect}")
            values = np.asarray(sorted(rows, key=lambda r: r[0]), dtype=float)
            grid, truth, mean = values.T
            if np.any(np.diff(grid) <= 0):
                raise ValueError(f"Duplicate grid points for {method}/{effect}")
            if not np.allclose(grid[[0, -1]], [0, 1], atol=1e-10, rtol=0):
                raise ValueError(f"The grid for {method}/{effect} must span [0, 1]")
            if reference_grid is None:
                reference_grid, reference_truth = grid, truth
            elif (
                grid.shape != reference_grid.shape
                or not np.allclose(grid, reference_grid, atol=1e-12, rtol=0)
                or not np.allclose(truth, reference_truth, atol=1e-10, rtol=0)
            ):
                raise ValueError(f"Inconsistent grid or truth across methods for {effect}")
            curves[effect][method] = (grid, truth, mean)
    return curves


def plot_curves(
    curves: dict, output_dir: Path, dpi: int = PNG_DPI, formats: tuple = OUTPUT_FORMATS
) -> list[Path]:
    """Draw the saved truth and posterior mean curves without smoothing."""
    if dpi <= 0 or not formats or any(f not in {"pdf", "png", "svg"} for f in formats):
        raise ValueError("Use a positive dpi and formats chosen from pdf, png, svg")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.1))
        try:
            for ax, effect, title in zip(axes, EFFECTS, TITLES):
                x, truth, _ = curves[effect]["joint"]
                ax.plot(x, truth, color="black", linewidth=2.65, label="Truth", zorder=3)
                for method in METHODS:
                    mx, _, mean = curves[effect][method]
                    label, color, linestyle, width = LINE_STYLES[method]
                    ax.plot(
                        mx,
                        mean,
                        label=label,
                        color=color,
                        linestyle=linestyle,
                        linewidth=width,
                        zorder=4,
                    )
                ax.set_title(title, loc="left", pad=8)
                ax.set_xlabel(r"State $t$", labelpad=4)
                ax.set_ylabel(r"$\mathrm{" + effect + r"}(t)$", labelpad=4)
                ax.set_xlim(0, 1)
                ax.set_xticks(np.linspace(0, 1, 6))
                ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction="out", pad=4)
                ax.grid(False)
                all_y = np.concatenate([truth] + [curves[effect][m][2] for m in METHODS])
                span = float(np.ptp(all_y))
                padding = span * 0.055 if span > 0 else max(abs(float(all_y[0])), 1) * 0.055
                ax.set_ylim(float(all_y.min()) - padding, float(all_y.max()) + padding)
                step, fmt = (0.25, "%.2f") if effect == "TE" else (0.2, "%.1f")
                ax.yaxis.set_major_locator(MultipleLocator(step))
                ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))

            handles, labels = axes[0].get_legend_handles_labels()
            legend = fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.015),
                ncol=5,
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
