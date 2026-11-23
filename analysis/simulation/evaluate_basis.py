"""Compute recovery for the Fourier-truth experiment and plot the effects for Figure S3."""

#!/usr/bin/env python3
from __future__ import annotations


import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from blafm.effects import effect_coefficient_mean
import pandas as pd
from blafm.mcmc import basis_design


# ---------------------------------------------------------------------
# Journal-friendly grayscale plotting style
# ---------------------------------------------------------------------
plt.style.use("default")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


REPLICATE_PATTERN = re.compile(r"replicate_(\d+)$")
EFFECT_ORDER = ("ME", "DE", "TE")
TARGET_ORDER = (
    "RISE_ME",
    "RISE_DE",
    "RISE_TE",
    "IMSE_g",
    "IMSE_h",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate paired K=4,5,6 basis-robustness fits and create "
            "the main-text LaTeX table and PDF figure."
        )
    )
    parser.add_argument("--replicates", type=int, nargs="+", help="Use these paired replicate IDs.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/basis"),
        help="Root directory containing replicate_* folders.",
    )
    parser.add_argument(
        "--K-values",
        nargs="+",
        type=int,
        default=list(range(4, 10)),
        help="Basis dimensions to evaluate. Default: 4 5 6 7 8 9.",
    )
    parser.add_argument(
        "--fit-pattern",
        default="fit_K_{K}",
        help=(
            "Fit-directory pattern relative to each replicate. "
            "It must contain the placeholder {K}. "
            "Default: fit_K_{K}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <root>/summary.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for basis_effect_curves_main.pdf/png. "
            "Default: the output directory."
        ),
    )
    parser.add_argument(
        "--expected-replicates",
        type=int,
        default=None,
        help="Optional required number of complete paired replicates.",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=3,
        help="B-spline degree. Default: 3.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Resolution of the PNG copy. The PDF remains vector based.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def integrate(values: np.ndarray, grid: np.ndarray, axis: int = -1) -> np.ndarray:
    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, grid, axis=axis)
    return np.trapz(values, grid, axis=axis)


def mcse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def discover_replicates(root: Path) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for path in sorted(root.glob("replicate_*")):
        if not path.is_dir():
            continue
        match = REPLICATE_PATTERN.fullmatch(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return found


def fit_dir_for(replicate_dir: Path, fit_pattern: str, k: int) -> Path:
    try:
        name = fit_pattern.format(K=k)
    except KeyError as exc:
        raise ValueError("--fit-pattern must contain the placeholder {K}.") from exc
    return replicate_dir / name


def required_files(
    replicate_dir: Path,
    fit_pattern: str,
    k_values: list[int],
) -> list[Path]:
    data_dir = replicate_dir / "generated_data"
    files = [
        data_dir / "grid.json",
        data_dir / "g_grid.json",
        data_dir / "h_grid.json",
        data_dir / "effect_truth.json",
    ]
    for k in k_values:
        files.append(fit_dir_for(replicate_dir, fit_pattern, k) / "posterior_set.json")
    return files


def find_complete_replicates(
    root: Path,
    fit_pattern: str,
    k_values: list[int],
) -> list[tuple[int, Path]]:
    all_replicates = discover_replicates(root)
    if not all_replicates:
        raise FileNotFoundError(f"No replicate_* directories were found under {root}.")

    complete: list[tuple[int, Path]] = []
    incomplete: list[tuple[int, list[Path]]] = []

    for replicate, replicate_dir in all_replicates:
        missing = [
            path
            for path in required_files(replicate_dir, fit_pattern, k_values)
            if not path.is_file()
        ]
        if missing:
            incomplete.append((replicate, missing))
        else:
            complete.append((replicate, replicate_dir))

    if incomplete:
        print("\nIncomplete replicates excluded from every K:")
        for replicate, missing in incomplete:
            print(f"  replicate_{replicate:03d}: {len(missing)} missing file(s)")
            for path in missing:
                print(f"    - {path}")

    if not complete:
        raise FileNotFoundError("No replicate contains generated truth and every requested K fit.")
    return complete


def normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def extract_effect_truth(effect_truth: Any, effect: str) -> np.ndarray:
    """
    Accept common key variants such as ME, me, ME_truth,
    mediated_effect, direct_effect, and total_effect.
    """
    if not isinstance(effect_truth, dict):
        raise TypeError("effect_truth.json must contain a JSON object.")

    normalized = {normalize_key(str(key)): key for key in effect_truth}
    candidates = {
        "ME": (
            "me",
            "metruth",
            "mediatedeffect",
            "mediatedeffecttruth",
            "indirecteffect",
        ),
        "DE": (
            "de",
            "detruth",
            "directeffect",
            "directeffecttruth",
        ),
        "TE": (
            "te",
            "tetruth",
            "totaleffect",
            "totaleffecttruth",
        ),
    }[effect]

    for candidate in candidates:
        if candidate in normalized:
            return np.asarray(effect_truth[normalized[candidate]], dtype=float).squeeze()

    raise KeyError(
        f"Could not find {effect} in effect_truth.json. Available keys: {list(effect_truth)}"
    )


def as_vector(value: Any, k: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).squeeze()
    if array.ndim != 1 or array.size != k:
        raise ValueError(
            f"{name} has shape {np.asarray(value).shape}; expected a vector of length {k}."
        )
    return array


def as_matrix(value: Any, k: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).squeeze()
    if array.shape != (k, k):
        raise ValueError(f"{name} has shape {np.asarray(value).shape}; expected ({k}, {k}).")
    return array


def load_fit_config(fit_dir: Path) -> dict[str, Any]:
    config_file = fit_dir / "fit_config.json"
    return read_json(config_file) if config_file.is_file() else {}


def posterior_mean(
    posterior: dict[str, Any],
    key: str,
    burnin: int | None,
) -> np.ndarray:
    """
    Prefer the saved posterior mean KEY_HAT. If absent, reconstruct it
    from the retained chain. Index 0 is the initial state.
    """
    hat_key = f"{key}_HAT"
    if hat_key in posterior:
        return np.asarray(posterior[hat_key], dtype=float)

    if key not in posterior:
        raise KeyError(f"posterior_set.json contains neither {hat_key} nor {key}.")
    if burnin is None:
        raise ValueError(f"{hat_key} is missing and fit_config.json does not provide burnin.")

    chain = np.asarray(posterior[key], dtype=float)
    start = int(burnin) + 1
    if start >= chain.shape[0]:
        raise ValueError(f"Cannot summarize {key}: burnin={burnin}, chain length={chain.shape[0]}.")
    return np.mean(chain[start:], axis=0)


def orient_truth_matrix(
    value: Any,
    n_donors: int,
    n_grid: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape == (n_donors, n_grid):
        return array
    if array.shape == (n_grid, n_donors):
        warnings.warn(f"Transposing {name} from {array.shape}.")
        return array.T
    raise ValueError(
        f"{name} has shape {array.shape}; expected "
        f"({n_donors}, {n_grid}) or ({n_grid}, {n_donors})."
    )


def interpolate_curve(
    source_grid: np.ndarray,
    source_values: np.ndarray,
    target_grid: np.ndarray,
) -> np.ndarray:
    source_grid = np.asarray(source_grid, dtype=float)
    source_values = np.asarray(source_values, dtype=float)
    if np.array_equal(source_grid, target_grid):
        return source_values
    return np.interp(target_grid, source_grid, source_values)


def evaluate_one_fit(
    data_dir: Path,
    fit_dir: Path,
    k: int,
    degree: int,
) -> tuple[dict[str, float], dict[str, np.ndarray], np.ndarray]:
    grid = np.asarray(read_json(data_dir / "grid.json"), dtype=float).squeeze()
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError(f"Invalid grid in {data_dir / 'grid.json'}.")
    if np.any(np.diff(grid) <= 0):
        raise ValueError("grid.json must be strictly increasing.")

    effect_truth_raw = read_json(data_dir / "effect_truth.json")
    truth = {effect: extract_effect_truth(effect_truth_raw, effect) for effect in EFFECT_ORDER}
    for effect, values in truth.items():
        if values.shape != grid.shape:
            raise ValueError(f"Truth {effect} has shape {values.shape}; expected {grid.shape}.")

    posterior = read_json(fit_dir / "posterior_set.json")
    config = load_fit_config(fit_dir)
    burnin = config.get("burnin")

    a_hat = as_vector(posterior_mean(posterior, "A", burnin), k, "A_HAT")
    b_hat = as_matrix(posterior_mean(posterior, "B", burnin), k, "B_HAT")
    c_hat = as_vector(posterior_mean(posterior, "C", burnin), k, "C_HAT")

    phi = basis_design(grid, k=k, degree=degree)
    ba_hat = effect_coefficient_mean(posterior, burnin)["ME"]
    curves = {
        "ME": phi @ ba_hat,
        "DE": phi @ c_hat,
        "TE": phi @ (ba_hat + c_hat),
    }

    metrics: dict[str, float] = {}
    for effect in EFFECT_ORDER:
        squared_error = (curves[effect] - truth[effect]) ** 2
        metrics[f"RISE_{effect}"] = float(np.sqrt(integrate(squared_error, grid)))

    if "alpha_pred" not in posterior or "beta_pred" not in posterior:
        raise KeyError(
            f"{fit_dir / 'posterior_set.json'} must contain alpha_pred "
            "and beta_pred for trajectory IMSE."
        )

    alpha_pred = np.asarray(posterior["alpha_pred"], dtype=float)
    beta_pred = np.asarray(posterior["beta_pred"], dtype=float)
    if alpha_pred.ndim != 2 or alpha_pred.shape[1] != k:
        raise ValueError(f"alpha_pred has shape {alpha_pred.shape}; expected (n, {k}).")
    if beta_pred.shape != alpha_pred.shape:
        raise ValueError(f"beta_pred has shape {beta_pred.shape}; expected {alpha_pred.shape}.")

    n_donors = alpha_pred.shape[0]
    g_true = orient_truth_matrix(
        read_json(data_dir / "g_grid.json"),
        n_donors=n_donors,
        n_grid=grid.size,
        name="g_grid",
    )
    h_true = orient_truth_matrix(
        read_json(data_dir / "h_grid.json"),
        n_donors=n_donors,
        n_grid=grid.size,
        name="h_grid",
    )

    g_hat = alpha_pred @ phi.T
    h_hat = beta_pred @ phi.T
    donor_ise_g = integrate((g_hat - g_true) ** 2, grid, axis=1)
    donor_ise_h = integrate((h_hat - h_true) ** 2, grid, axis=1)
    metrics["IMSE_g"] = float(np.mean(donor_ise_g))
    metrics["IMSE_h"] = float(np.mean(donor_ise_h))

    for effect in EFFECT_ORDER:
        estimate = float(integrate(curves[effect], grid))
        target = float(integrate(truth[effect], grid))
        metrics["I" + effect + "_estimate"] = estimate
        metrics["I" + effect + "_truth"] = target
        metrics["I" + effect + "_error"] = estimate - target

    combined_curves = {
        **{f"truth_{effect}": truth[effect] for effect in EFFECT_ORDER},
        **{f"estimate_{effect}": curves[effect] for effect in EFFECT_ORDER},
    }
    return metrics, combined_curves, grid


def summarize_metrics(replicate_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for k in sorted(replicate_metrics["K"].unique()):
        subset = replicate_metrics[replicate_metrics["K"] == k]
        row: dict[str, Any] = {
            "K": int(k),
            "n_replicates": int(len(subset)),
        }
        for target in TARGET_ORDER:
            values = subset[target].to_numpy(dtype=float)
            row[target] = float(np.mean(values))
            row[f"MCSE_{target}"] = mcse(values)
        rows.append(row)
    return pd.DataFrame(rows)


def average_pointwise_curves(
    curve_records: list[dict[str, Any]],
    common_grid: np.ndarray,
    k_values: list[int],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[int, dict[str, np.ndarray]]]:
    truth_stacks: dict[str, list[np.ndarray]] = {effect: [] for effect in EFFECT_ORDER}
    estimate_stacks: dict[int, dict[str, list[np.ndarray]]] = {
        k: {effect: [] for effect in EFFECT_ORDER} for k in k_values
    }

    for record in curve_records:
        source_grid = record["grid"]
        k = record["K"]
        for effect in EFFECT_ORDER:
            truth_stacks[effect].append(
                interpolate_curve(
                    source_grid,
                    record[f"truth_{effect}"],
                    common_grid,
                )
            )
            estimate_stacks[k][effect].append(
                interpolate_curve(
                    source_grid,
                    record[f"estimate_{effect}"],
                    common_grid,
                )
            )

    truth_mean = {
        effect: np.mean(np.stack(values, axis=0), axis=0) for effect, values in truth_stacks.items()
    }
    estimate_mean = {
        k: {
            effect: np.mean(np.stack(values, axis=0), axis=0)
            for effect, values in effect_values.items()
        }
        for k, effect_values in estimate_stacks.items()
    }

    frame_data: dict[str, np.ndarray] = {"pseudotime": common_grid}
    for effect in EFFECT_ORDER:
        frame_data[f"truth_{effect}"] = truth_mean[effect]
        for k in k_values:
            frame_data[f"K{k}_{effect}"] = estimate_mean[k][effect]

    return pd.DataFrame(frame_data), truth_mean, estimate_mean


def get_basis_line_styles(k_values: list[int]) -> dict[int, dict[str, Any]]:
    """
    Grayscale styles following the reference plotting format.

    The truth is drawn separately as a thick black dashed line.
    Fitted curves use progressively lighter grayscale tones and
    distinct line types so that the figure remains interpretable
    after black-and-white printing.
    """
    default_styles = {
        4: {
            "color": "0.28",
            "linestyle": (0, (7, 2.5)),
            "linewidth": 1.8,
        },
        5: {
            "color": "0.46",
            "linestyle": (0, (5, 2, 1.2, 2)),
            "linewidth": 1.7,
        },
        6: {
            "color": "0.64",
            "linestyle": (0, (1.2, 2.0)),
            "linewidth": 1.9,
        },
    }

    fallback_patterns = [
        (0, (8, 2)),
        (0, (5, 2, 1, 2)),
        (0, (3, 2)),
        "--",
    ]

    styles: dict[int, dict[str, Any]] = {}
    for index, k in enumerate(k_values):
        if k in default_styles:
            styles[k] = default_styles[k]
        else:
            styles[k] = {
                "color": str(min(0.15 + 0.12 * index, 0.65)),
                "linestyle": fallback_patterns[index % len(fallback_patterns)],
                "linewidth": 1.7,
            }
    return styles


def _plot_basis_curve_panel(
    axis: plt.Axes,
    grid: np.ndarray,
    truth_curve: np.ndarray,
    estimate_mean: dict[int, dict[str, np.ndarray]],
    k_values: list[int],
    effect: str,
    ylabel: str,
    title: str,
) -> tuple[list[Any], list[str]]:
    styles = get_basis_line_styles(k_values)

    handles: list[Any] = []
    labels: list[str] = []

    (truth_line,) = axis.plot(
        grid,
        truth_curve,
        color="0.00",
        linestyle="-",
        linewidth=2.5,
        label="Truth",
        zorder=4,
    )
    handles.append(truth_line)
    labels.append("Truth")

    for k in k_values:
        style = styles[k]
        (fitted_line,) = axis.plot(
            grid,
            estimate_mean[k][effect],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            label=rf"$K={k}$",
            zorder=3,
        )
        handles.append(fitted_line)
        labels.append(rf"$K={k}$")

    axis.axhline(
        0.0,
        color="0.78",
        linestyle=(0, (2, 2)),
        linewidth=0.75,
        zorder=0,
    )
    axis.set_xlim(float(grid.min()), float(grid.max()))
    axis.set_xlabel(r"Pseudotime $t$")
    axis.set_ylabel(ylabel)
    axis.set_title(title, pad=6)
    axis.tick_params(axis="both", which="major", length=3)
    axis.margins(x=0.0, y=0.08)
    axis.set_facecolor("white")

    return handles, labels


def make_figure(
    grid: np.ndarray,
    truth_mean: dict[str, np.ndarray],
    estimate_mean: dict[int, dict[str, np.ndarray]],
    k_values: list[int],
    pdf_path: Path,
    png_path: Path,
    dpi: int,
) -> None:
    """
    Create a single-row, three-panel grayscale figure using the same
    visual language as the supplied sensitivity-analysis plotting code:
    serif fonts, outward ticks, open top/right spines, distinct
    black/gray line styles, zero reference lines, and one common legend.
    """
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(13.2, 3.6),
    )
    figure.patch.set_facecolor("white")

    panel_settings = {
        "ME": {
            "ylabel": r"$ME(t)$",
            "title": "Mediation effect",
        },
        "DE": {
            "ylabel": r"$DE(t)$",
            "title": "Direct effect",
        },
        "TE": {
            "ylabel": r"$TE(t)$",
            "title": "Total effect",
        },
    }

    common_handles: list[Any] | None = None
    common_labels: list[str] | None = None

    for axis, effect in zip(axes, EFFECT_ORDER):
        settings = panel_settings[effect]
        handles, labels = _plot_basis_curve_panel(
            axis=axis,
            grid=grid,
            truth_curve=truth_mean[effect],
            estimate_mean=estimate_mean,
            k_values=k_values,
            effect=effect,
            ylabel=settings["ylabel"],
            title=settings["title"],
        )
        if common_handles is None:
            common_handles = handles
            common_labels = labels

    if common_handles is None or common_labels is None:
        raise RuntimeError("No legend handles were generated.")

    figure.legend(
        common_handles,
        common_labels,
        loc="lower center",
        ncol=len(common_labels),
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        handlelength=2.8,
        columnspacing=1.6,
    )

    figure.subplots_adjust(
        left=0.065,
        right=0.985,
        top=0.88,
        bottom=0.25,
        wspace=0.32,
    )

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    k_values = sorted(set(args.K_values))
    if not k_values:
        raise ValueError("At least one K value is required.")

    output_dir = args.output_dir.resolve() if args.output_dir is not None else root / "summary"
    figure_dir = args.figure_dir.resolve() if args.figure_dir is not None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    replicates = find_complete_replicates(
        root=root,
        fit_pattern=args.fit_pattern,
        k_values=k_values,
    )
    if args.replicates:
        selected = set(args.replicates)
        replicates = [(r, d) for r, d in replicates if r in selected]
        missing = selected.difference(r for r, _ in replicates)
        if missing:
            raise ValueError(f"Requested replicates are incomplete: {sorted(missing)}")
    if args.expected_replicates is not None and len(replicates) != args.expected_replicates:
        raise ValueError(
            f"Found {len(replicates)} complete paired replicates, but "
            f"--expected-replicates={args.expected_replicates}."
        )

    print(f"\nEvaluating {len(replicates)} complete paired replicates for K={k_values}.")

    metric_rows: list[dict[str, Any]] = []
    curve_records: list[dict[str, Any]] = []
    common_grid: np.ndarray | None = None

    for replicate, replicate_dir in replicates:
        data_dir = replicate_dir / "generated_data"

        for k in k_values:
            fit_dir = fit_dir_for(replicate_dir, args.fit_pattern, k)
            metrics, curves, grid = evaluate_one_fit(
                data_dir=data_dir,
                fit_dir=fit_dir,
                k=k,
                degree=args.degree,
            )

            metric_rows.append(
                {
                    "replicate": replicate,
                    "K": k,
                    **metrics,
                }
            )
            curve_records.append(
                {
                    "replicate": replicate,
                    "K": k,
                    "grid": grid,
                    **curves,
                }
            )

            if common_grid is None:
                common_grid = grid.copy()

        print(f"  completed replicate_{replicate:03d}")

    if common_grid is None:
        raise RuntimeError("No evaluation grid was loaded.")

    replicate_metrics = pd.DataFrame(metric_rows)
    summary = summarize_metrics(replicate_metrics)

    pointwise, truth_mean, estimate_mean = average_pointwise_curves(
        curve_records=curve_records,
        common_grid=common_grid,
        k_values=k_values,
    )

    csv_float_format = "%.17g"
    replicate_metrics.to_csv(
        output_dir / "basis_robustness_replicate_metrics.csv",
        index=False,
        float_format=csv_float_format,
    )
    summary.to_csv(
        output_dir / "basis_robustness_summary.csv",
        index=False,
        float_format=csv_float_format,
    )
    pointwise.to_csv(
        output_dir / "basis_effect_curves_pointwise.csv",
        index=False,
        float_format=csv_float_format,
    )

    pdf_path = figure_dir / "basis_effect_curves_main.pdf"
    png_path = figure_dir / "basis_effect_curves_main.png"
    make_figure(
        grid=common_grid,
        truth_mean=truth_mean,
        estimate_mean=estimate_mean,
        k_values=k_values,
        pdf_path=pdf_path,
        png_path=png_path,
        dpi=args.dpi,
    )

    print(f"Saved full-fit basis recovery in {output_dir}")


if __name__ == "__main__":
    main()
