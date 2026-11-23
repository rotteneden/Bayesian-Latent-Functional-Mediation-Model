"""Compute paired path, effect-curve and donor-trajectory recovery for Table 1 and Table S1."""

#!/usr/bin/env python3
from __future__ import annotations


import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from blafm.effects import effect_coefficient_mean
import pandas as pd
from blafm.mcmc import basis_design


# ---------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute finite-sample recovery from paired replicate fits."
    )
    parser.add_argument("--replicates", type=int, nargs="+", help="Use these paired replicate IDs.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/finite"),
        help="Root directory containing replicate_*/n_*/.",
    )
    parser.add_argument(
        "--sample-sizes",
        nargs="+",
        type=int,
        default=[200, 500, 1000],
        help="Sample sizes to evaluate. Default: 200 500 1000.",
    )
    parser.add_argument(
        "--fit-dir-name",
        default="fit_K5",
        help="Fit directory inside each n_* directory. Default: fit_K5.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <root>/summary.",
    )
    parser.add_argument(
        "--expected-replicates",
        type=int,
        default=None,
        help="Optional expected number of complete paired replicates, e.g. 30.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------
def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def integrate(values: np.ndarray, grid: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerical integral over the supplied grid."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, grid, axis=axis)
    return np.trapz(values, grid, axis=axis)


def mcse(values: np.ndarray) -> float:
    """Simulation Monte Carlo standard error of a replicate mean."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def as_vector(value: Any, expected_length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).squeeze()
    if array.ndim != 1 or array.size != expected_length:
        raise ValueError(
            f"{name} has shape {np.asarray(value).shape}; "
            f"expected a vector of length {expected_length}."
        )
    return array


def as_matrix(value: Any, expected_shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).squeeze()
    if array.shape != expected_shape:
        raise ValueError(f"{name} has shape {np.asarray(value).shape}; expected {expected_shape}.")
    return array


def posterior_mean(
    posterior: dict[str, Any],
    key: str,
    *,
    burnin: int | None = None,
) -> np.ndarray:
    """
    Prefer the saved *_HAT value. If absent, calculate the posterior mean
    from the full chain, accounting for the initial state at index 0.
    """
    hat_key = f"{key}_HAT"
    if hat_key in posterior:
        return np.asarray(posterior[hat_key], dtype=float)

    if key not in posterior:
        raise KeyError(f"Neither {hat_key} nor {key} is present in posterior_set.json.")

    chain = np.asarray(posterior[key], dtype=float)
    if burnin is None:
        raise ValueError(f"{hat_key} is missing, so burn-in is required to summarize {key}.")

    start = burnin + 1  # index 0 is the initial state
    if start >= chain.shape[0]:
        raise ValueError(f"Cannot summarize {key}: burnin={burnin}, chain length={chain.shape[0]}.")
    return np.mean(chain[start:], axis=0)


# ---------------------------------------------------------------------
# Replicate discovery and validation
# ---------------------------------------------------------------------
REPLICATE_PATTERN = re.compile(r"replicate_(\d+)$")


def discover_replicates(root: Path) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for path in sorted(root.glob("replicate_*")):
        if not path.is_dir():
            continue
        match = REPLICATE_PATTERN.fullmatch(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return found


def required_paths(
    replicate_dir: Path,
    sample_size: int,
    fit_dir_name: str,
) -> tuple[Path, Path, Path]:
    scenario_dir = replicate_dir / f"n_{sample_size}"
    data_dir = scenario_dir / "generated_data"
    fit_dir = scenario_dir / fit_dir_name
    posterior_file = fit_dir / "posterior_set.json"
    return data_dir, fit_dir, posterior_file


def complete_replicates(
    root: Path,
    sample_sizes: list[int],
    fit_dir_name: str,
) -> list[tuple[int, Path]]:
    """
    Use only replicates that contain every requested sample size.
    This preserves the paired nested-replicate design.
    """
    all_replicates = discover_replicates(root)
    complete: list[tuple[int, Path]] = []
    incomplete: list[str] = []

    for replicate, replicate_dir in all_replicates:
        missing: list[str] = []
        for n in sample_sizes:
            data_dir, _, posterior_file = required_paths(replicate_dir, n, fit_dir_name)
            needed = [
                data_dir / "path_truth.json",
                data_dir / "effect_truth.json",
                data_dir / "grid.json",
                data_dir / "alpha.json",
                data_dir / "beta.json",
                posterior_file,
            ]
            missing.extend(str(path) for path in needed if not path.is_file())

        if missing:
            incomplete.append(f"replicate_{replicate:03d}: {len(missing)} required file(s) missing")
        else:
            complete.append((replicate, replicate_dir))

    if incomplete:
        print("\nIncomplete replicates excluded from all sample sizes:")
        for item in incomplete:
            print(f"  - {item}")

    if not complete:
        raise FileNotFoundError(
            "No complete paired replicates were found for all requested sample sizes."
        )

    return complete


# ---------------------------------------------------------------------
# Metric calculations
# ---------------------------------------------------------------------
def load_fit_config(fit_dir: Path) -> dict[str, Any]:
    path = fit_dir / "fit_config.json"
    return read_json(path) if path.is_file() else {}


def path_metrics_for_one_fit(
    data_dir: Path,
    fit_dir: Path,
    posterior: dict[str, Any],
) -> list[dict[str, float | int | str]]:
    """
    Return component-level estimates and errors.

    Bias is a repeated-simulation property, so it is calculated later across
    replicates rather than within a single replicate.
    """
    truth = read_json(data_dir / "path_truth.json")
    k = int(truth.get("basis_dimension", 4))

    config = load_fit_config(fit_dir)
    burnin = config.get("burnin")

    a_true = as_vector(truth["A"], k, "true A")
    b_true = np.diag(as_matrix(truth["B"], (k, k), "true B"))
    c_true = as_vector(truth["C"], k, "true C")

    a_hat = as_vector(posterior_mean(posterior, "A", burnin=burnin), k, "estimated A")
    b_hat_matrix = as_matrix(
        posterior_mean(posterior, "B", burnin=burnin),
        (k, k),
        "estimated B",
    )
    b_hat = np.diag(b_hat_matrix)
    c_hat = as_vector(posterior_mean(posterior, "C", burnin=burnin), k, "estimated C")

    results: list[dict[str, float | int | str]] = []
    for block, estimate, target in (
        ("A", a_hat, a_true),
        ("diag(B)", b_hat, b_true),
        ("C", c_hat, c_true),
    ):
        for component in range(k):
            error = float(estimate[component] - target[component])
            results.append(
                {
                    "parameter_block": block,
                    "component": component + 1,
                    "estimate": float(estimate[component]),
                    "truth": float(target[component]),
                    "error": error,
                    "absolute_error": abs(error),
                    "squared_error": error**2,
                }
            )
    return results


def load_or_reconstruct_trajectory_truth(
    data_dir: Path,
    grid: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    g_grid_file = data_dir / "g_grid.json"
    h_grid_file = data_dir / "h_grid.json"

    if g_grid_file.is_file() and h_grid_file.is_file():
        g_true = np.asarray(read_json(g_grid_file), dtype=float)
        h_true = np.asarray(read_json(h_grid_file), dtype=float)
    else:
        alpha_true = np.asarray(read_json(data_dir / "alpha.json"), dtype=float)
        beta_true = np.asarray(read_json(data_dir / "beta.json"), dtype=float)
        g_true = alpha_true @ phi.T
        h_true = beta_true @ phi.T

    expected_grid_size = grid.size
    if g_true.ndim != 2 or g_true.shape[1] != expected_grid_size:
        raise ValueError(f"g truth has shape {g_true.shape}; expected (n, {expected_grid_size}).")
    if h_true.shape != g_true.shape:
        raise ValueError(f"g and h truth shapes differ: {g_true.shape} versus {h_true.shape}.")
    return g_true, h_true


def function_metrics_for_one_fit(
    data_dir: Path,
    fit_dir: Path,
    posterior: dict[str, Any],
) -> list[dict[str, float | str]]:
    effect_truth = read_json(data_dir / "effect_truth.json")
    path_truth = read_json(data_dir / "path_truth.json")

    grid = np.asarray(read_json(data_dir / "grid.json"), dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError(f"Invalid evaluation grid in {data_dir / 'grid.json'}.")

    k = int(path_truth.get("basis_dimension", 4))
    degree = int(path_truth.get("spline_degree", 3))
    phi = basis_design(grid, k, degree)

    config = load_fit_config(fit_dir)
    burnin = config.get("burnin")

    a_hat = as_vector(posterior_mean(posterior, "A", burnin=burnin), k, "estimated A")
    b_hat = as_matrix(
        posterior_mean(posterior, "B", burnin=burnin),
        (k, k),
        "estimated B",
    )
    c_hat = as_vector(posterior_mean(posterior, "C", burnin=burnin), k, "estimated C")

    mediated_coefficients = effect_coefficient_mean(posterior, burnin)["ME"]
    direct_coefficients = c_hat
    total_coefficients = mediated_coefficients + direct_coefficients

    estimated_effects = {
        "ME(t)": phi @ mediated_coefficients,
        "DE(t)": phi @ direct_coefficients,
        "TE(t)": phi @ total_coefficients,
    }
    true_effects = {
        "ME(t)": np.asarray(effect_truth["ME"], dtype=float),
        "DE(t)": np.asarray(effect_truth["DE"], dtype=float),
        "TE(t)": np.asarray(effect_truth["TE"], dtype=float),
    }

    rows: list[dict[str, float | str]] = []
    for target in ("ME(t)", "DE(t)", "TE(t)"):
        estimate = estimated_effects[target]
        truth = true_effects[target]
        if estimate.shape != truth.shape:
            raise ValueError(
                f"{target} estimate/truth shapes differ: {estimate.shape} versus {truth.shape}."
            )
        rise = float(np.sqrt(integrate((estimate - truth) ** 2, grid)))
        rows.append(
            {
                "target": target,
                "error_metric": "RISE",
                "mean_error": rise,
            }
        )

    if "alpha_pred" not in posterior or "beta_pred" not in posterior:
        raise KeyError(
            "posterior_set.json must contain alpha_pred and beta_pred for donor-trajectory IMSE."
        )

    alpha_pred = np.asarray(posterior["alpha_pred"], dtype=float)
    beta_pred = np.asarray(posterior["beta_pred"], dtype=float)
    if alpha_pred.ndim != 2 or alpha_pred.shape[1] != k:
        raise ValueError(f"alpha_pred has shape {alpha_pred.shape}; expected (n, {k}).")
    if beta_pred.shape != alpha_pred.shape:
        raise ValueError(f"beta_pred has shape {beta_pred.shape}; expected {alpha_pred.shape}.")

    g_true, h_true = load_or_reconstruct_trajectory_truth(data_dir, grid, phi)
    if g_true.shape[0] != alpha_pred.shape[0]:
        raise ValueError(
            f"Trajectory truth has {g_true.shape[0]} donors, but alpha_pred "
            f"has {alpha_pred.shape[0]}."
        )

    g_hat = alpha_pred @ phi.T
    h_hat = beta_pred @ phi.T

    donor_ise_g = integrate((g_hat - g_true) ** 2, grid, axis=1)
    donor_ise_h = integrate((h_hat - h_true) ** 2, grid, axis=1)

    rows.extend(
        [
            {
                "target": "g_i(t)",
                "error_metric": "IMSE",
                "mean_error": float(np.mean(donor_ise_g)),
            },
            {
                "target": "h_i(t)",
                "error_metric": "IMSE",
                "mean_error": float(np.mean(donor_ise_h)),
            },
        ]
    )
    return rows


# ---------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------
def _path_block_statistics(errors: np.ndarray) -> tuple[float, float]:
    """
    Standard block-level repeated-simulation summaries.

    errors has shape (R, K):
      mean absolute bias = mean_k | mean_r(error_rk) |
      RMSE = sqrt( mean_{r,k}(error_rk^2) )
    """
    errors = np.asarray(errors, dtype=float)
    if errors.ndim != 2:
        raise ValueError(f"errors must be two-dimensional; received {errors.shape}.")
    component_bias = np.mean(errors, axis=0)
    mean_absolute_bias = float(np.mean(np.abs(component_bias)))
    block_rmse = float(np.sqrt(np.mean(errors**2)))
    return mean_absolute_bias, block_rmse


def _jackknife_mcse_for_path_metrics(
    errors: np.ndarray,
) -> tuple[float, float]:
    """
    Delete-one-master-replicate jackknife MCSE for the two nonlinear
    block-level summaries.
    """
    errors = np.asarray(errors, dtype=float)
    r = errors.shape[0]
    if r < 2:
        return float("nan"), float("nan")

    leave_one_out = np.empty((r, 2), dtype=float)
    for index in range(r):
        reduced = np.delete(errors, index, axis=0)
        leave_one_out[index] = _path_block_statistics(reduced)

    center = np.mean(leave_one_out, axis=0)
    variance = (r - 1) / r * np.sum((leave_one_out - center) ** 2, axis=0)
    standard_errors = np.sqrt(variance)
    return float(standard_errors[0]), float(standard_errors[1])


def summarize_path_metrics(path_replicates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    block_order = ["A", "diag(B)", "C"]

    for n in sorted(path_replicates["sample_size"].unique()):
        for block in block_order:
            subset = path_replicates[
                (path_replicates["sample_size"] == n)
                & (path_replicates["parameter_block"] == block)
            ].copy()

            pivot = (
                subset.pivot(
                    index="replicate",
                    columns="component",
                    values="error",
                )
                .sort_index(axis=0)
                .sort_index(axis=1)
            )

            if pivot.isna().any().any():
                raise ValueError(f"Missing component error for n={n}, block={block}.")

            errors = pivot.to_numpy(dtype=float)
            mean_absolute_bias, block_rmse = _path_block_statistics(errors)
            mcse_abs_bias, mcse_rmse = _jackknife_mcse_for_path_metrics(errors)

            rows.append(
                {
                    "sample_size": int(n),
                    "parameter_block": block,
                    "mean_absolute_bias": mean_absolute_bias,
                    "mcse_mean_absolute_bias": mcse_abs_bias,
                    "rmse": block_rmse,
                    "mcse_rmse": mcse_rmse,
                    "n_replicates": int(errors.shape[0]),
                    "n_components": int(errors.shape[1]),
                }
            )
    return pd.DataFrame(rows)


def summarize_function_metrics(function_replicates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target_order = ["ME(t)", "DE(t)", "TE(t)", "g_i(t)", "h_i(t)"]

    for n in sorted(function_replicates["sample_size"].unique()):
        for target in target_order:
            subset = function_replicates[
                (function_replicates["sample_size"] == n)
                & (function_replicates["target"] == target)
            ]
            values = subset["mean_error"].to_numpy(dtype=float)
            metric_values = subset["error_metric"].unique()
            if len(metric_values) != 1:
                raise ValueError(
                    f"Target {target}, n={n} has inconsistent metric labels: "
                    f"{metric_values.tolist()}."
                )
            rows.append(
                {
                    "sample_size": int(n),
                    "target": target,
                    "error_metric": metric_values[0],
                    "mean_error": float(np.mean(values)),
                    "mcse_mean_error": mcse(values),
                    "n_replicates": int(len(subset)),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    sample_sizes = sorted(set(args.sample_sizes))
    output_dir = args.output_dir.resolve() if args.output_dir is not None else root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    replicates = complete_replicates(root, sample_sizes, args.fit_dir_name)
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

    print(f"\nEvaluating {len(replicates)} complete paired replicates for n={sample_sizes}.")

    path_rows: list[dict[str, Any]] = []
    function_rows: list[dict[str, Any]] = []

    for replicate, replicate_dir in replicates:
        for n in sample_sizes:
            data_dir, fit_dir, posterior_file = required_paths(replicate_dir, n, args.fit_dir_name)
            posterior = read_json(posterior_file)

            for metric_row in path_metrics_for_one_fit(data_dir, fit_dir, posterior):
                path_rows.append(
                    {
                        "replicate": replicate,
                        "sample_size": n,
                        **metric_row,
                    }
                )

            for metric_row in function_metrics_for_one_fit(data_dir, fit_dir, posterior):
                function_rows.append(
                    {
                        "replicate": replicate,
                        "sample_size": n,
                        **metric_row,
                    }
                )

        print(f"  completed replicate_{replicate:03d}")

    path_replicates = pd.DataFrame(path_rows)
    function_replicates = pd.DataFrame(function_rows)

    path_summary = summarize_path_metrics(path_replicates)
    function_summary = summarize_function_metrics(function_replicates)

    csv_float_format = "%.17g"

    path_replicates.to_csv(
        output_dir / "finite_sample_path_replicate_metrics.csv",
        index=False,
        float_format=csv_float_format,
    )
    function_replicates.to_csv(
        output_dir / "finite_sample_function_replicate_metrics.csv",
        index=False,
        float_format=csv_float_format,
    )
    path_summary.to_csv(
        output_dir / "finite_sample_path_summary.csv",
        index=False,
        float_format=csv_float_format,
    )
    function_summary.to_csv(
        output_dir / "finite_sample_function_summary.csv",
        index=False,
        float_format=csv_float_format,
    )

    print(f"Saved paired finite-sample summaries in {output_dir}")


if __name__ == "__main__":
    main()
