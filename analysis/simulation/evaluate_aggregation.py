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

REPLICATE_PATTERN = re.compile(r"replicate_(\d+)$")
EFFECTS = ("ME", "DE", "TE")
FUNCTIONAL_METHODS = (
    ("BLaFM", "latent"),
    ("Oracle-pseudotime mediation", "oracle"),
)
SCALAR_EFFECT_PREFERRED_KEYS = {
    "Indirect effect": ["Indirect effect", "indirect"],
    "Direct effect": ["Direct effect", "direct"],
    "Total effect": [
        "Total effect (decomposition)",
        "Total effect",
        "total_decomposition",
        "Total effect (marginal model)",
        "total_marginal",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate aggregation-stress results and write the integrated-effect table and pointwise curves."
    )
    parser.add_argument("--replicates", type=int, nargs="+", help="Use these paired replicate IDs.")
    parser.add_argument("--root", type=Path, default=Path("outputs/aggregation"))
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--latent-dir-name", type=str, default=None)
    parser.add_argument("--oracle-dir-name", type=str, default=None)
    parser.add_argument("--scalar-dir-name", type=str, default="fit_scalar")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--expected-replicates", type=int, default=None)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--digits", type=int, default=3)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def integrate(values: np.ndarray, grid: np.ndarray, axis: int = -1) -> np.ndarray:
    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, grid, axis=axis)
    return np.trapz(values, grid, axis=axis)


def discover_replicates(root: Path):
    replicates = []
    for path in sorted(root.glob("replicate_*")):
        if path.is_dir():
            match = REPLICATE_PATTERN.fullmatch(path.name)
            if match:
                replicates.append((int(match.group(1)), path))
    return replicates


def latent_dir_name(k: int, explicit: str | None) -> str:
    return explicit if explicit is not None else f"fit_K{k}"


def oracle_dir_name(k: int, explicit: str | None) -> str:
    return explicit if explicit is not None else f"fit_K{k}_oracleT"


def required_files_for_replicate(
    replicate_dir: Path, n: int, k: int, latent_name: str, oracle_name: str, scalar_name: str
):
    base = replicate_dir / f"n_{n}"
    data_dir = base / "generated_data"
    scalar_dir = base / scalar_name
    files = [
        data_dir / "grid.json",
        data_dir / "effect_truth.json",
        base / latent_name / "posterior_set.json",
        base / oracle_name / "posterior_set.json",
    ]
    scalar_csv = scalar_dir / "scalar_effects.csv"
    scalar_json = scalar_dir / "scalar_mediation_fit.json"
    if scalar_csv.is_file():
        files.append(scalar_csv)
    elif scalar_json.is_file():
        files.append(scalar_json)
    else:
        files.append(scalar_csv)
    return files


def find_complete_replicates(
    root: Path, n: int, k: int, latent_name: str, oracle_name: str, scalar_name: str
):
    all_replicates = discover_replicates(root)
    if not all_replicates:
        raise FileNotFoundError(f"No replicate_* directories were found under {root}.")
    complete = []
    incomplete = []
    for replicate, replicate_dir in all_replicates:
        missing = [
            path
            for path in required_files_for_replicate(
                replicate_dir, n, k, latent_name, oracle_name, scalar_name
            )
            if not path.is_file()
        ]
        if missing:
            incomplete.append((replicate, missing))
        else:
            complete.append((replicate, replicate_dir))
    if incomplete:
        print("\nIncomplete replicates excluded from all methods:")
        for replicate, missing in incomplete:
            print(f"  replicate_{replicate:03d}: {len(missing)} missing file(s)")
            for path in missing:
                print(f"    - {path}")
    if not complete:
        raise FileNotFoundError(
            "No replicate contains complete latent, oracle, and scalar outputs."
        )
    return complete


def extract_effect_truth(effect_truth: Any, effect: str) -> np.ndarray:
    if not isinstance(effect_truth, dict):
        raise TypeError("effect_truth.json must contain a JSON object.")
    normalized = {normalize_key(str(key)): key for key in effect_truth}
    candidates = {
        "ME": ("me", "metruth", "mediatedeffect", "mediatedeffecttruth", "indirecteffect"),
        "DE": ("de", "detruth", "directeffect", "directeffecttruth"),
        "TE": ("te", "tetruth", "totaleffect", "totaleffecttruth"),
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


def posterior_mean(posterior: dict[str, Any], key: str, burnin: int | None) -> np.ndarray:
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


def evaluate_functional_fit(data_dir: Path, fit_dir: Path, k: int, degree: int):
    grid = np.asarray(read_json(data_dir / "grid.json"), dtype=float).squeeze()
    effect_truth_raw = read_json(data_dir / "effect_truth.json")
    truth = {effect: extract_effect_truth(effect_truth_raw, effect) for effect in EFFECTS}
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
    integrated_estimates = {
        "IME": float(integrate(curves["ME"], grid)),
        "IDE": float(integrate(curves["DE"], grid)),
        "ITE": float(integrate(curves["TE"], grid)),
        "truth_IME": float(integrate(truth["ME"], grid)),
        "truth_IDE": float(integrate(truth["DE"], grid)),
        "truth_ITE": float(integrate(truth["TE"], grid)),
    }
    curve_record = {
        **{f"truth_{effect}": truth[effect] for effect in EFFECTS},
        **{f"estimate_{effect}": curves[effect] for effect in EFFECTS},
    }
    return integrated_estimates, curve_record, grid


def scalar_estimates_from_csv(path: Path):
    frame = pd.read_csv(path)
    effect_map = {str(row["effect"]).strip(): float(row["estimate"]) for _, row in frame.iterrows()}
    extracted = {}
    for display_name, keys in SCALAR_EFFECT_PREFERRED_KEYS.items():
        value = None
        for key in keys:
            if key in effect_map:
                value = effect_map[key]
                break
        if value is None:
            raise KeyError(
                f"{path} does not contain the scalar effect {display_name!r}. Available effects: {list(effect_map)}"
            )
        extracted[display_name] = float(value)
    return extracted


def scalar_estimates_from_json(path: Path):
    payload = read_json(path)
    effects = payload["effects"]
    extracted = {}
    for display_name, keys in SCALAR_EFFECT_PREFERRED_KEYS.items():
        value = None
        for key in keys:
            if key in effects:
                value = effects[key]
                break
        if value is None:
            raise KeyError(
                f"{path} does not contain the scalar effect {display_name!r}. Available effect keys: {list(effects)}"
            )
        extracted[display_name] = float(value)
    return extracted


def load_scalar_estimates(scalar_dir: Path):
    csv_path = scalar_dir / "scalar_effects.csv"
    if csv_path.is_file():
        return scalar_estimates_from_csv(csv_path)
    json_path = scalar_dir / "scalar_mediation_fit.json"
    if json_path.is_file():
        return scalar_estimates_from_json(json_path)
    raise FileNotFoundError(
        f"Neither scalar_effects.csv nor scalar_mediation_fit.json was found in {scalar_dir}."
    )


def summary_with_jackknife(estimates: np.ndarray, truths: np.ndarray):
    estimates = np.asarray(estimates, dtype=float)
    truths = np.asarray(truths, dtype=float)
    errors = estimates - truths
    mean_estimate = float(np.mean(estimates))
    bias = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors**2)))
    n = estimates.size
    if n < 2:
        return {
            "mean_estimate": mean_estimate,
            "mcse_mean_estimate": float("nan"),
            "bias": bias,
            "mcse_bias": float("nan"),
            "rmse": rmse,
            "mcse_rmse": float("nan"),
        }
    loo_stats = np.empty((n, 3), dtype=float)
    for index in range(n):
        est_reduced = np.delete(estimates, index)
        truth_reduced = np.delete(truths, index)
        err_reduced = est_reduced - truth_reduced
        loo_stats[index, 0] = np.mean(est_reduced)
        loo_stats[index, 1] = np.mean(err_reduced)
        loo_stats[index, 2] = np.sqrt(np.mean(err_reduced**2))
    center = np.mean(loo_stats, axis=0)
    variance = (n - 1) / n * np.sum((loo_stats - center) ** 2, axis=0)
    mcse_mean, mcse_bias, mcse_rmse = np.sqrt(variance)
    return {
        "mean_estimate": mean_estimate,
        "mcse_mean_estimate": float(mcse_mean),
        "bias": bias,
        "mcse_bias": float(mcse_bias),
        "rmse": rmse,
        "mcse_rmse": float(mcse_rmse),
    }


def summarize_integrated(replicate_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for display_name, method_key in FUNCTIONAL_METHODS:
        for effect in ("IME", "IDE", "ITE"):
            subset = replicate_metrics[replicate_metrics["method"] == method_key]
            stats = summary_with_jackknife(
                estimates=subset[effect].to_numpy(dtype=float),
                truths=subset[f"truth_{effect}"].to_numpy(dtype=float),
            )
            rows.append(
                {
                    "method_display": display_name,
                    "effect_display": effect,
                    "true_value": float(np.mean(subset[f"truth_{effect}"])),
                    **stats,
                    "n_replicates": int(len(subset)),
                }
            )
    scalar_subset = replicate_metrics[replicate_metrics["method"] == "scalar"]
    scalar_mappings = [
        ("Indirect effect", "scalar_indirect", "truth_IME"),
        ("Direct effect", "scalar_direct", "truth_IDE"),
        ("Total effect", "scalar_total", "truth_ITE"),
    ]
    for display_name, estimate_column, truth_column in scalar_mappings:
        stats = summary_with_jackknife(
            estimates=scalar_subset[estimate_column].to_numpy(dtype=float),
            truths=scalar_subset[truth_column].to_numpy(dtype=float),
        )
        rows.append(
            {
                "method_display": "Donor-average scalar mediation",
                "effect_display": display_name,
                "true_value": float(np.mean(scalar_subset[truth_column])),
                **stats,
                "n_replicates": int(len(scalar_subset)),
            }
        )
    return pd.DataFrame(rows)


def interpolate_curve(
    source_grid: np.ndarray, source_values: np.ndarray, target_grid: np.ndarray
) -> np.ndarray:
    if np.array_equal(source_grid, target_grid):
        return np.asarray(source_values, dtype=float)
    return np.interp(target_grid, source_grid, source_values)


def average_pointwise_curves(records, common_grid):
    truth_stacks = {effect: [] for effect in EFFECTS}
    method_stacks = {
        "latent": {effect: [] for effect in EFFECTS},
        "oracle": {effect: [] for effect in EFFECTS},
    }
    scalar_values = {"ME": [], "DE": [], "TE": []}
    for record in records:
        source_grid = record["grid"]
        method = record["method"]
        if method in method_stacks:
            for effect in EFFECTS:
                truth_stacks[effect].append(
                    interpolate_curve(source_grid, record[f"truth_{effect}"], common_grid)
                )
                method_stacks[method][effect].append(
                    interpolate_curve(source_grid, record[f"estimate_{effect}"], common_grid)
                )
        elif method == "scalar":
            scalar_values["ME"].append(float(record["scalar_indirect"]))
            scalar_values["DE"].append(float(record["scalar_direct"]))
            scalar_values["TE"].append(float(record["scalar_total"]))
    truth_mean = {
        effect: np.mean(np.stack(values, axis=0), axis=0) for effect, values in truth_stacks.items()
    }
    method_mean = {
        method: {
            effect: np.mean(np.stack(effect_values, axis=0), axis=0)
            for effect, effect_values in method_dict.items()
        }
        for method, method_dict in method_stacks.items()
    }
    scalar_mean = {effect: float(np.mean(values)) for effect, values in scalar_values.items()}
    frame = pd.DataFrame({"pseudotime": common_grid})
    for effect in EFFECTS:
        frame[f"truth_{effect}"] = truth_mean[effect]
        frame[f"latent_{effect}"] = method_mean["latent"][effect]
        frame[f"oracle_{effect}"] = method_mean["oracle"][effect]
        frame[f"scalar_{effect}"] = np.repeat(scalar_mean[effect], common_grid.size)
    return frame, truth_mean, method_mean, scalar_mean


def format_estimate_mcse(estimate: float, mcse: float, digits: int) -> str:
    if not np.isfinite(estimate):
        return "--"
    if not np.isfinite(mcse):
        return f"${estimate:.{digits}f}$"
    return rf"${estimate:.{digits}f}\;({mcse:.{digits}f})$"


def format_true_value(value: float, digits: int) -> str:
    rounded_1 = round(value, 1)
    if abs(value - rounded_1) < 10 ** (-(digits + 1)):
        return f"${rounded_1:.1f}$"
    return f"${value:.{digits}f}$"


def make_latex_table(summary: pd.DataFrame, digits: int, n_replicates: int) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Recovery of integrated effect summaries in the targeted",
        r"aggregation-stress experiment. Results are based on",
        rf"$R_{{\mathrm{{MC}}}}={n_replicates}$ paired simulation replicates. Simulation Monte",
        r"Carlo standard errors are shown in parentheses.}",
        r"\label{tab:aggregation_integrated_main}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Method",
        r"& Effect",
        r"& True value",
        r"& Mean estimate",
        r"& Bias",
        r"& RMSE \\",
        r"\midrule",
    ]
    method_blocks = [
        ("BLaFM", ["IME", "IDE", "ITE"]),
        ("Oracle-pseudotime mediation", ["IME", "IDE", "ITE"]),
        ("Donor-average scalar mediation", ["Indirect effect", "Direct effect", "Total effect"]),
    ]
    for method_index, (method_name, effects) in enumerate(method_blocks):
        subset = summary[summary["method_display"] == method_name]
        for row_index, effect in enumerate(effects):
            row = subset[subset["effect_display"] == effect]
            row = row.iloc[0]
            method_cell = method_name if row_index == 0 else ""
            lines.append(
                " & ".join(
                    [
                        method_cell,
                        effect,
                        format_true_value(float(row["true_value"]), digits),
                        format_estimate_mcse(
                            float(row["mean_estimate"]), float(row["mcse_mean_estimate"]), digits
                        ),
                        format_estimate_mcse(float(row["bias"]), float(row["mcse_bias"]), digits),
                        format_estimate_mcse(float(row["rmse"]), float(row["mcse_rmse"]), digits),
                    ]
                )
                + r" \\"
            )
        if method_index < len(method_blocks) - 1:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "",
            r"\vspace{4pt}",
            "",
            r"\begin{minipage}{0.94\textwidth}",
            r"\footnotesize",
            r"\textit{Note:}",
            r"For the latent and oracle functional models, IME, IDE, and ITE denote",
            r"the integrated mediated, direct, and total effects, respectively. The",
            r"scalar indirect, direct, and total effects were evaluated against the",
            r"same integrated functional targets. Under the aggregation design, the",
            r"population scalar projection values are $0$, $0.5$, and $0.5$,",
            r"respectively.",
            r"\end{minipage}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Root directory does not exist: {root}")
    latent_name = latent_dir_name(args.K, args.latent_dir_name)
    oracle_name = oracle_dir_name(args.K, args.oracle_dir_name)
    scalar_name = args.scalar_dir_name
    output_dir = args.output_dir.resolve() if args.output_dir is not None else root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    replicates = find_complete_replicates(
        root, args.n, args.K, latent_name, oracle_name, scalar_name
    )
    if args.replicates:
        selected = set(args.replicates)
        replicates = [(r, d) for r, d in replicates if r in selected]
        missing = selected.difference(r for r, _ in replicates)
        if missing:
            raise ValueError(f"Requested replicates are incomplete: {sorted(missing)}")
    if args.expected_replicates is not None and len(replicates) != args.expected_replicates:
        raise ValueError(
            f"Found {len(replicates)} complete paired replicates, but --expected-replicates={args.expected_replicates}."
        )
    print(
        f"\nEvaluating {len(replicates)} complete paired replicates for aggregation-stress, n={args.n}, K={args.K}."
    )
    metric_rows = []
    curve_records = []
    common_grid = None
    for replicate, replicate_dir in replicates:
        base = replicate_dir / f"n_{args.n}"
        data_dir = base / "generated_data"
        latent_dir = base / latent_name
        oracle_dir = base / oracle_name
        scalar_dir = base / scalar_name
        latent_metrics, latent_curves, grid = evaluate_functional_fit(
            data_dir, latent_dir, args.K, args.degree
        )
        oracle_metrics, oracle_curves, _ = evaluate_functional_fit(
            data_dir, oracle_dir, args.K, args.degree
        )
        scalar_estimates = load_scalar_estimates(scalar_dir)
        if common_grid is None:
            common_grid = grid.copy()
        metric_rows.append({"replicate": replicate, "method": "latent", **latent_metrics})
        metric_rows.append({"replicate": replicate, "method": "oracle", **oracle_metrics})
        metric_rows.append(
            {
                "replicate": replicate,
                "method": "scalar",
                "IME": np.nan,
                "IDE": np.nan,
                "ITE": np.nan,
                "truth_IME": latent_metrics["truth_IME"],
                "truth_IDE": latent_metrics["truth_IDE"],
                "truth_ITE": latent_metrics["truth_ITE"],
                "scalar_indirect": scalar_estimates["Indirect effect"],
                "scalar_direct": scalar_estimates["Direct effect"],
                "scalar_total": scalar_estimates["Total effect"],
            }
        )
        curve_records.append(
            {"replicate": replicate, "method": "latent", "grid": grid, **latent_curves}
        )
        curve_records.append(
            {"replicate": replicate, "method": "oracle", "grid": grid, **oracle_curves}
        )
        curve_records.append(
            {
                "replicate": replicate,
                "method": "scalar",
                "grid": grid,
                "scalar_indirect": scalar_estimates["Indirect effect"],
                "scalar_direct": scalar_estimates["Direct effect"],
                "scalar_total": scalar_estimates["Total effect"],
            }
        )
        print(f"  completed replicate_{replicate:03d}")
    replicate_metrics = pd.DataFrame(metric_rows)
    summary = summarize_integrated(replicate_metrics)
    pointwise, _, _, _ = average_pointwise_curves(curve_records, common_grid)
    replicate_metrics.to_csv(
        output_dir / "aggregation_integrated_replicate_metrics.csv",
        index=False,
        float_format="%.17g",
    )
    summary.to_csv(
        output_dir / "aggregation_integrated_summary.csv", index=False, float_format="%.17g"
    )
    pointwise.to_csv(
        output_dir / "aggregation_effect_curves_pointwise.csv", index=False, float_format="%.17g"
    )
    latex = make_latex_table(summary, args.digits, len(replicates))
    latex_path = output_dir / "table_aggregation_integrated_main.tex"
    latex_path.write_text(latex, encoding="utf-8")
    print(f"Saved integrated effects and pointwise curves in {output_dir}")


if __name__ == "__main__":
    main()
