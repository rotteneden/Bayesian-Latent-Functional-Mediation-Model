"""Fit donor-average scalar mediation with a donor bootstrap and a marginal total-effect benchmark."""

#!/usr/bin/env python3
from __future__ import annotations


import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from analysis.fit_model import digest, atomic_json


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit donor-average product-of-coefficients scalar mediation."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        help="Donor-level bootstrap resamples; use 0 to skip. Default: 2000.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level. Default: 0.95.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-bootstrap-draws", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true")
    group.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def serialize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize(item) for item in value]
    return value


def validate_data(
    x_raw: Any,
    y_raw: Any,
    z_raw: Any,
    c_raw: Any | None,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    if not isinstance(x_raw, list) or not isinstance(y_raw, list):
        raise TypeError("X.json and Y.json must contain donor-level lists.")

    z = np.asarray(z_raw, dtype=float).reshape(-1)
    n = len(x_raw)
    if len(y_raw) != n or z.size != n:
        raise ValueError(f"Inconsistent donor counts: X={n}, Y={len(y_raw)}, Z={z.size}.")
    if n < 4:
        raise ValueError("At least four donors are required.")
    if not np.all(np.isfinite(z)) or np.var(z) <= 0:
        raise ValueError("Z must be finite and have positive variance.")

    x: list[np.ndarray] = []
    y: list[np.ndarray] = []
    cell_counts = np.empty(n, dtype=int)

    for donor in range(n):
        x_i = np.asarray(x_raw[donor], dtype=float).reshape(-1)
        y_i = np.asarray(y_raw[donor], dtype=float).reshape(-1)
        if x_i.size == 0 or y_i.size == 0:
            raise ValueError(f"Donor {donor} has no cells.")
        if x_i.size != y_i.size:
            raise ValueError(f"Donor {donor}: len(X_i)={x_i.size}, len(Y_i)={y_i.size}.")
        if not np.all(np.isfinite(x_i)) or not np.all(np.isfinite(y_i)):
            raise ValueError(f"Donor {donor} contains non-finite X or Y values.")
        x.append(x_i)
        y.append(y_i)
        cell_counts[donor] = x_i.size

    if c_raw is not None:
        c_loaded = np.asarray(c_raw, dtype=int).reshape(-1)
        if c_loaded.size != n or not np.array_equal(c_loaded, cell_counts):
            raise ValueError("C.json does not match donor counts derived from X/Y.")

    return x, y, z, cell_counts


def fit_linear_model(
    response: np.ndarray,
    design: np.ndarray,
    term_names: list[str],
    confidence: float,
) -> dict[str, Any]:
    response = np.asarray(response, dtype=float).reshape(-1)
    design = np.asarray(design, dtype=float)
    n, p = design.shape

    if response.size != n:
        raise ValueError("Response and design have incompatible row counts.")
    if len(term_names) != p:
        raise ValueError("term_names does not match the design columns.")
    if np.linalg.matrix_rank(design) < p:
        raise np.linalg.LinAlgError(f"Rank-deficient design for {term_names}.")

    xtx_inv = np.linalg.inv(design.T @ design)
    coefficients = xtx_inv @ design.T @ response
    fitted = design @ coefficients
    residuals = response - fitted
    dof = n - p
    if dof <= 0:
        raise ValueError("Residual degrees of freedom must be positive.")

    rss = float(residuals @ residuals)
    tss = float(((response - np.mean(response)) ** 2).sum())
    sigma2 = rss / dof

    covariance_ols = sigma2 * xtx_inv
    se_ols = np.sqrt(np.maximum(np.diag(covariance_ols), 0.0))

    leverage = np.sum((design @ xtx_inv) * design, axis=1)
    denominator = np.maximum(1.0 - leverage, np.finfo(float).eps)
    hc3_scale = (residuals / denominator) ** 2
    covariance_hc3 = xtx_inv @ (design.T @ (design * hc3_scale[:, None])) @ xtx_inv
    se_hc3 = np.sqrt(np.maximum(np.diag(covariance_hc3), 0.0))

    alpha = 1.0 - confidence
    critical = float(student_t.ppf(1.0 - alpha / 2.0, dof))

    def inference(standard_errors: np.ndarray) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index, term in enumerate(term_names):
            estimate = float(coefficients[index])
            se = float(standard_errors[index])
            if se > 0 and np.isfinite(se):
                statistic = estimate / se
                p_value = float(2.0 * student_t.sf(abs(statistic), df=dof))
                lower = estimate - critical * se
                upper = estimate + critical * se
            else:
                statistic = p_value = lower = upper = float("nan")
            rows.append(
                {
                    "term": term,
                    "estimate": estimate,
                    "standard_error": se,
                    "t_value": float(statistic),
                    "p_value": float(p_value),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                }
            )
        return rows

    r_squared = float(1.0 - rss / tss) if tss > 0 else float("nan")
    adjusted_r_squared = (
        float(1.0 - (1.0 - r_squared) * (n - 1) / dof) if np.isfinite(r_squared) else float("nan")
    )

    return {
        "n": n,
        "n_parameters": p,
        "degrees_of_freedom": dof,
        "terms": term_names,
        "coefficients": coefficients,
        "rss": rss,
        "residual_standard_deviation": float(math.sqrt(sigma2)),
        "r_squared": r_squared,
        "adjusted_r_squared": adjusted_r_squared,
        "covariance_ols": covariance_ols,
        "covariance_hc3": covariance_hc3,
        "inference_ols": inference(se_ols),
        "inference_hc3": inference(se_hc3),
    }


def coefficient_by_term(model: dict[str, Any], term: str) -> float:
    return float(model["coefficients"][model["terms"].index(term)])


def fit_once(
    z: np.ndarray,
    x_bar: np.ndarray,
    y_bar: np.ndarray,
    confidence: float,
) -> dict[str, Any]:
    ones = np.ones(z.size, dtype=float)

    mediator_model = fit_linear_model(
        x_bar,
        np.column_stack([ones, z]),
        ["Intercept", "Z"],
        confidence,
    )
    outcome_model = fit_linear_model(
        y_bar,
        np.column_stack([ones, z, x_bar]),
        ["Intercept", "Z", "Xbar"],
        confidence,
    )
    total_model = fit_linear_model(
        y_bar,
        np.column_stack([ones, z]),
        ["Intercept", "Z"],
        confidence,
    )

    a = coefficient_by_term(mediator_model, "Z")
    b = coefficient_by_term(outcome_model, "Xbar")
    c_prime = coefficient_by_term(outcome_model, "Z")
    c_marginal = coefficient_by_term(total_model, "Z")
    indirect = a * b
    total_decomposition = indirect + c_prime

    return {
        "mediator_model": mediator_model,
        "outcome_model": outcome_model,
        "marginal_total_model": total_model,
        "path_coefficients": {
            "a": a,
            "b": b,
            "c_prime": c_prime,
            "c_marginal": c_marginal,
        },
        "effects": {
            "indirect": indirect,
            "direct": c_prime,
            "total_decomposition": total_decomposition,
            "total_marginal": c_marginal,
            "decomposition_minus_marginal": total_decomposition - c_marginal,
        },
    }


def bootstrap_effects(
    z: np.ndarray,
    x_bar: np.ndarray,
    y_bar: np.ndarray,
    n_bootstrap: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = [
        "a",
        "b",
        "c_prime",
        "c_marginal",
        "indirect",
        "direct",
        "total_decomposition",
        "total_marginal",
    ]
    if n_bootstrap == 0:
        return pd.DataFrame(columns=columns), {
            "requested": 0,
            "successful": 0,
            "failed_rank_deficient": 0,
            "confidence": confidence,
            "effects": {},
        }

    rows: list[dict[str, float]] = []
    failed = 0
    n = z.size

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        try:
            fit = fit_once(z[indices], x_bar[indices], y_bar[indices], confidence)
        except (np.linalg.LinAlgError, ValueError):
            failed += 1
            continue

        path = fit["path_coefficients"]
        effect = fit["effects"]
        rows.append(
            {
                "a": path["a"],
                "b": path["b"],
                "c_prime": path["c_prime"],
                "c_marginal": path["c_marginal"],
                "indirect": effect["indirect"],
                "direct": effect["direct"],
                "total_decomposition": effect["total_decomposition"],
                "total_marginal": effect["total_marginal"],
            }
        )

    draws = pd.DataFrame(rows, columns=columns)
    if draws.empty:
        raise RuntimeError("Every bootstrap resample was rank deficient.")

    alpha = 1.0 - confidence
    summaries: dict[str, dict[str, float]] = {}
    for name in ("indirect", "direct", "total_decomposition", "total_marginal"):
        values = draws[name].to_numpy(dtype=float)
        lower, upper = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
        p_nonpositive = float(np.mean(values <= 0.0))
        p_nonnegative = float(np.mean(values >= 0.0))
        summaries[name] = {
            "bootstrap_mean": float(np.mean(values)),
            "bootstrap_standard_error": float(np.std(values, ddof=1)),
            "ci_lower_percentile": float(lower),
            "ci_upper_percentile": float(upper),
            "probability_positive": float(np.mean(values > 0.0)),
            "probability_negative": float(np.mean(values < 0.0)),
            "two_sided_sign_tail_probability": float(
                min(1.0, 2.0 * min(p_nonpositive, p_nonnegative))
            ),
        }

    return draws, {
        "requested": int(n_bootstrap),
        "successful": int(len(draws)),
        "failed_rank_deficient": int(failed),
        "confidence": confidence,
        "interval_method": "donor-level nonparametric percentile bootstrap",
        "effects": summaries,
    }


def model_for_json(model: dict[str, Any]) -> dict[str, Any]:
    return {
        key: model[key]
        for key in (
            "n",
            "n_parameters",
            "degrees_of_freedom",
            "terms",
            "coefficients",
            "rss",
            "residual_standard_deviation",
            "r_squared",
            "adjusted_r_squared",
            "covariance_ols",
            "covariance_hc3",
            "inference_ols",
            "inference_hc3",
        )
    }


def coefficient_table(models: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, model in models.items():
        ols = {row["term"]: row for row in model["inference_ols"]}
        hc3 = {row["term"]: row for row in model["inference_hc3"]}
        for term in model["terms"]:
            row_ols = ols[term]
            row_hc3 = hc3[term]
            rows.append(
                {
                    "model": model_name,
                    "term": term,
                    "estimate": row_ols["estimate"],
                    "ols_standard_error": row_ols["standard_error"],
                    "ols_t_value": row_ols["t_value"],
                    "ols_p_value": row_ols["p_value"],
                    "ols_ci_lower": row_ols["ci_lower"],
                    "ols_ci_upper": row_ols["ci_upper"],
                    "hc3_standard_error": row_hc3["standard_error"],
                    "hc3_t_value": row_hc3["t_value"],
                    "hc3_p_value": row_hc3["p_value"],
                    "hc3_ci_lower": row_hc3["ci_lower"],
                    "hc3_ci_upper": row_hc3["ci_upper"],
                    "n": model["n"],
                    "degrees_of_freedom": model["degrees_of_freedom"],
                    "r_squared": model["r_squared"],
                    "adjusted_r_squared": model["adjusted_r_squared"],
                }
            )
    return pd.DataFrame(rows)


def effects_table(
    point_effects: dict[str, float],
    bootstrap_summary: dict[str, Any],
) -> pd.DataFrame:
    mapping = [
        ("Indirect effect", "indirect"),
        ("Direct effect", "direct"),
        ("Total effect (decomposition)", "total_decomposition"),
        ("Total effect (marginal model)", "total_marginal"),
    ]
    bootstrap = bootstrap_summary.get("effects", {})
    rows: list[dict[str, Any]] = []
    for label, key in mapping:
        summary = bootstrap.get(key, {})
        rows.append(
            {
                "effect": label,
                "estimate": point_effects[key],
                "bootstrap_standard_error": summary.get("bootstrap_standard_error", np.nan),
                "ci_lower_percentile": summary.get("ci_lower_percentile", np.nan),
                "ci_upper_percentile": summary.get("ci_upper_percentile", np.nan),
                "probability_positive": summary.get("probability_positive", np.nan),
                "probability_negative": summary.get("probability_negative", np.nan),
                "two_sided_sign_tail_probability": summary.get(
                    "two_sided_sign_tail_probability", np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_arguments()
    if args.bootstrap < 0:
        raise ValueError("--bootstrap must be nonnegative.")
    if not (0.0 < args.confidence < 1.0):
        raise ValueError("--confidence must lie strictly between 0 and 1.")

    output_file = args.output / "scalar_mediation_fit.json"
    required = ("X", "Y", "Z")
    missing = [name for name in required if not (args.data / f"{name}.json").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input files in {args.data}: {', '.join(missing)}")

    sources = list(required) + (["C"] if (args.data / "C.json").is_file() else [])
    request = dict(
        bootstrap=args.bootstrap,
        seed=args.seed,
        confidence=args.confidence,
        data_sha256={name: digest(args.data / (name + ".json")) for name in sources},
        driver_sha256=digest(__file__),
        save_bootstrap_draws=args.save_bootstrap_draws,
    )
    config_file = args.output / "fit_config.json"
    if output_file.exists() or config_file.exists():
        if args.resume and config_file.is_file():
            cached = read_json(config_file)
            files = cached.get("products", {})
            if (
                files
                and cached.get("request") == request
                and all(
                    (args.output / name).is_file() and digest(args.output / name) == value
                    for name, value in files.items()
                )
            ):
                print(f"Using completed scalar fit in {args.output}")
                return
        if not args.overwrite:
            raise FileExistsError(f"Incomplete or different fit at {args.output}.")
    x_raw = read_json(args.data / "X.json")
    y_raw = read_json(args.data / "Y.json")
    z_raw = read_json(args.data / "Z.json")
    c_file = args.data / "C.json"
    c_raw = read_json(c_file) if c_file.is_file() else None

    x, y, z, cell_counts = validate_data(x_raw, y_raw, z_raw, c_raw)
    x_bar = np.asarray([np.mean(item) for item in x], dtype=float)
    y_bar = np.asarray([np.mean(item) for item in y], dtype=float)

    fit = fit_once(z, x_bar, y_bar, args.confidence)
    rng = np.random.default_rng(args.seed)
    bootstrap_draws, bootstrap_summary = bootstrap_effects(
        z,
        x_bar,
        y_bar,
        args.bootstrap,
        args.confidence,
        rng,
    )

    result = {
        "method": "donor-average scalar product-of-coefficients mediation",
        "n_donors": int(z.size),
        "total_cells": int(np.sum(cell_counts)),
        "cell_count_summary": {
            "minimum": int(np.min(cell_counts)),
            "median": float(np.median(cell_counts)),
            "mean": float(np.mean(cell_counts)),
            "maximum": int(np.max(cell_counts)),
        },
        "averaging": (
            "Each donor contributes one unweighted Xbar and Ybar; "
            "donors receive equal regression weight."
        ),
        "models": {
            "mediator_Xbar_on_Z": model_for_json(fit["mediator_model"]),
            "outcome_Ybar_on_Z_and_Xbar": model_for_json(fit["outcome_model"]),
            "marginal_total_Ybar_on_Z": model_for_json(fit["marginal_total_model"]),
        },
        "path_coefficients": fit["path_coefficients"],
        "effects": fit["effects"],
        "bootstrap": bootstrap_summary,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(serialize(result), file, ensure_ascii=False, indent=2)

    pd.DataFrame(
        {
            "donor_index": np.arange(z.size, dtype=int),
            "Z": z,
            "Xbar": x_bar,
            "Ybar": y_bar,
            "cell_count": cell_counts,
        }
    ).to_csv(args.output / "donor_average_data.csv", index=False)

    coefficient_table(
        {
            "M1: Xbar ~ Z": fit["mediator_model"],
            "M2: Ybar ~ Z + Xbar": fit["outcome_model"],
            "M3: Ybar ~ Z": fit["marginal_total_model"],
        }
    ).to_csv(args.output / "regression_coefficients.csv", index=False)

    effects_table(fit["effects"], bootstrap_summary).to_csv(
        args.output / "scalar_effects.csv", index=False
    )

    if args.save_bootstrap_draws and not bootstrap_draws.empty:
        bootstrap_draws.to_csv(args.output / "bootstrap_effect_draws.csv", index=False)

    config = {
        "data": str(args.data),
        "output": str(args.output),
        "bootstrap": args.bootstrap,
        "bootstrap_successful": bootstrap_summary["successful"],
        "confidence": args.confidence,
        "seed": args.seed,
        "save_bootstrap_draws": args.save_bootstrap_draws,
        "completed_at": datetime.now().astimezone().isoformat(),
    }
    products = [
        "scalar_mediation_fit.json",
        "donor_average_data.csv",
        "regression_coefficients.csv",
        "scalar_effects.csv",
    ]
    if args.save_bootstrap_draws and not bootstrap_draws.empty:
        products.append("bootstrap_effect_draws.csv")
    config.update(request=request, products={name: digest(args.output / name) for name in products})
    atomic_json(config_file, config)

    paths = fit["path_coefficients"]
    effects = fit["effects"]
    print("----- Donor-average scalar mediation -----")
    print(f"Donors: {z.size}")
    print(f"Total cells: {int(np.sum(cell_counts))}")
    print(f"a (Z -> Xbar):              {paths['a']:.8g}")
    print(f"b (Xbar -> Ybar | Z):       {paths['b']:.8g}")
    print(f"c' (Z -> Ybar | Xbar):      {paths['c_prime']:.8g}")
    print(f"c  (Z -> Ybar):             {paths['c_marginal']:.8g}")
    print(f"Indirect effect a*b:        {effects['indirect']:.8g}")
    print(f"Direct effect c':           {effects['direct']:.8g}")
    print(f"Total effect a*b+c':        {effects['total_decomposition']:.8g}")
    print(f"Marginal total effect c:    {effects['total_marginal']:.8g}")
    print(f"Decomposition check:        {effects['decomposition_minus_marginal']:.3e}")
    if args.bootstrap > 0:
        print(f"Bootstrap successful:       {bootstrap_summary['successful']}/{args.bootstrap}")
    print(f"Saved scalar fit to {output_file}")


if __name__ == "__main__":
    main()
