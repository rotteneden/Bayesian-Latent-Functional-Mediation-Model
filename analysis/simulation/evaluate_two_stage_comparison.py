#!/usr/bin/env python3
"""Evaluate paired K=5,n=1000 FMA fits. No frequentist coverage calculation."""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
from .fit_two_stage_comparison import (
    Data,
    K,
    N,
    DEGREE,
    KNOTS,
    METHODS,
    read_json,
    finite_array,
    sha256,
    effect_coefficients,
    mcmc,
    atomic_json,
    STAGE1_FOLDER,
    STAGE1_TRACE_KEYS,
    DEFAULT_OUTPUT,
    VERSION,
)

LABELS = {
    "joint": "BLaFM",
    "twostage_t": "Two-stage t",
    "twostage_coef": "Two-stage coefficients",
    "oracle_t": "Oracle t",
}


def integral(y, x, axis=-1):
    fn = getattr(np, "trapezoid", None)
    return (fn if fn is not None else np.trapz)(y, x=x, axis=axis)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_mcse(values):
    a = np.asarray(values, float)
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else None


def rmse_jackknife_mcse(errors):
    """Delete whole replicates, not individual coefficient components."""
    e = np.asarray(errors, float).reshape(len(errors), -1)
    mse = np.mean(e * e, axis=1)
    estimate, r = float(np.sqrt(mse.mean())), len(e)
    if r < 2:
        return estimate, None
    jk = np.sqrt(np.maximum(0.0, (mse.sum() - mse) / (r - 1)))
    return estimate, float(np.sqrt((r - 1) / r * np.sum((jk - jk.mean()) ** 2)))


def load_truth(data):
    d = data.directory
    pt, et = read_json(d / "path_truth.json"), read_json(d / "effect_truth.json")
    if pt.get("basis_dimension") != K or pt.get("spline_degree") != DEGREE:
        raise ValueError(f"{d}: truth must use K=5 and degree=3")
    grid = finite_array(read_json(d / "grid.json"), "grid")
    if (
        grid.ndim != 1
        or len(grid) < 3
        or np.any(np.diff(grid) <= 0)
        or abs(grid[0]) > 1e-12
        or abs(grid[-1] - 1) > 1e-12
    ):
        raise ValueError("Evaluation grid must strictly increase from 0 to 1")
    if "grid" in et and not np.allclose(et["grid"], grid, atol=1e-12, rtol=0):
        raise ValueError("grid.json and effect_truth grid disagree")
    phi = mcmc.bspline_basis_matrix(grid, KNOTS, DEGREE)
    a = finite_array(pt["A"], "truth A").reshape(K)
    bmat = finite_array(pt["B"], "truth B", (K, K))
    if not np.allclose(bmat, np.diag(np.diag(bmat)), atol=1e-12, rtol=0):
        raise ValueError("This recovery comparison requires diagonal true B")
    c = finite_array(pt["C"], "truth C").reshape(K)
    effects = {k: finite_array(et[k], "truth " + k, grid.shape) for k in ("ME", "DE", "TE")}
    expected = {"ME": phi @ (bmat @ a), "DE": phi @ c, "TE": phi @ (bmat @ a + c)}
    for name in effects:
        if not np.allclose(effects[name], expected[name], atol=1e-9, rtol=1e-7):
            raise ValueError(f"{d}: inconsistent effect_truth/path_truth for {name}")
        if "I" + name in et and not np.isclose(
            et["I" + name], integral(effects[name], grid), atol=1e-9, rtol=1e-7
        ):
            raise ValueError(f"{d}: integrated truth mismatch for {name}")
    alpha = finite_array(read_json(d / "alpha.json"), "true alpha", (N, K))
    beta = finite_array(read_json(d / "beta.json"), "true beta", (N, K))
    return grid, phi, effects, dict(A=a, b=np.diag(bmat), C=c), alpha, beta, data.true_t()


def load_fit(folder, data, rep, method):
    cfg = read_json(folder / "fit_config.json")
    req = cfg["request"]
    for name, val in (
        ("replicate", rep),
        ("method", method),
        ("K", K),
        ("n", N),
        ("version", VERSION),
    ):
        if req.get(name) != val:
            raise ValueError(f"{folder}: unexpected {name}")
    if req["observed_sha256"] != data.observed_hashes:
        raise ValueError(f"{folder}: observed data changed since fitting")
    if cfg["result_sha256"] != sha256(folder / "posterior.npz"):
        raise ValueError(f"{folder}: posterior checksum mismatch")
    with np.load(folder / "posterior.npz", allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    for name in ("A", "b", "C"):
        a = finite_array(out[name], name)
        if a.ndim != 2 or a.shape[1] != K or a.shape[0] < 2:
            raise ValueError(f"{folder}: invalid {name} draws")
    if not (out["A"].shape == out["b"].shape == out["C"].shape):
        raise ValueError("Population draw arrays are not aligned")
    # ALREADY post-burn-in. Do not discard burn-in a second time.
    return out, cfg


def load_stage1(folder, data, rep):
    cfg = read_json(folder / "fit_config.json")
    req = cfg["request"]
    for key, value in (
        ("replicate", rep),
        ("method", "stage1"),
        ("K", K),
        ("n", N),
        ("version", VERSION),
    ):
        if req.get(key) != value:
            raise ValueError(f"{folder}: unexpected {key}; v2 requires a new hierarchical Stage 1")
    if req["observed_sha256"] != data.observed_hashes:
        raise ValueError(f"{folder}: observed data changed since Stage 1")
    if req.get("stage1_prior", {}).get("model") != "independent_hierarchical_normal_diagonal":
        raise ValueError(f"{folder}: wrong Stage 1 prior model")
    checksum = sha256(folder / "posterior.npz")
    if cfg["result_sha256"] != checksum:
        raise ValueError(f"{folder}: Stage 1 checksum mismatch")
    with np.load(folder / "posterior.npz", allow_pickle=False) as z:
        traces = {key: finite_array(z[key], key).copy() for key in STAGE1_TRACE_KEYS}
    kept = req["iterations"] - req["burnin"]
    for key, trace in traces.items():
        if trace.shape != (kept, K) or kept < 2:
            raise ValueError(f"{folder}: invalid {key} trace shape")
        if key.startswith("v_") and np.any(trace <= 0):
            raise ValueError(f"{folder}: nonpositive population variance")
    return traces, cfg, checksum


def evaluate(args):
    outdir = args.output_root.resolve()
    reps = sorted(set(args.replicates))
    if not reps or min(reps) < 1:
        raise ValueError("replicates must be positive")
    missing = [
        str(outdir / f"replicate_{r:03d}" / m / f)
        for r in reps
        for m in (*METHODS, STAGE1_FOLDER)
        for f in ("posterior.npz", "fit_config.json")
        if not (outdir / f"replicate_{r:03d}" / m / f).is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing requested paired fits:\n" + "\n".join(missing[:20]))
    function_rows, integrated_rows, coefficient_rows, recovery_rows, timing_rows = (
        [],
        [],
        [],
        [],
        [],
    )
    hyperparameter_rows = []
    curve_store = {m: {e: [] for e in ("ME", "DE", "TE")} for m in METHODS}
    dataset_info, request_reference = [], {}
    grid_ref = truth_ref = None
    for rep in reps:
        data = Data(args.root / f"replicate_{rep:03d}" / "n_1000" / "generated_data")
        grid, phi, truths, path_truth, at, bt, tt = load_truth(data)
        if grid_ref is None:
            grid_ref, truth_ref = grid, truths
        elif not np.array_equal(grid, grid_ref) or any(
            not np.allclose(truths[e], truth_ref[e], atol=1e-10, rtol=0) for e in truths
        ):
            raise ValueError("Summary expects identical population truths/grid across replicates")
        basis_weights = integral(phi, grid, axis=0)
        s1folder = outdir / f"replicate_{rep:03d}" / STAGE1_FOLDER
        s1traces, s1cfg, s1hash = load_stage1(s1folder, data, rep)
        s1settings = {
            k: s1cfg["request"][k]
            for k in (
                "iterations",
                "burnin",
                "priors",
                "stage1_prior",
                "knots",
                "degree",
                "driver_sha256",
                "sampler_sha256",
                "version",
            )
        }
        if "stage1" in request_reference and s1settings != request_reference["stage1"]:
            raise ValueError("Stage 1 configuration changed across replicates")
        request_reference["stage1"] = s1settings
        for name, trace in s1traces.items():
            for k in range(K):
                x = trace[:, k]
                diagnostic = s1cfg["diagnostics"]["trace_diagnostics"][f"{name}[{k}]"]
                hyperparameter_rows.append(
                    {
                        "replicate": rep,
                        "parameter": name,
                        "component": k + 1,
                        "posterior_mean": float(x.mean()),
                        "posterior_sd": float(x.std(ddof=1)),
                        "CrI025": float(np.quantile(x, 0.025)),
                        "CrI975": float(np.quantile(x, 0.975)),
                        "ess_raw_single_chain": diagnostic["ess_raw_single_chain"],
                        "mcmc_mcse_mean": diagnostic["mcmc_mcse_mean"],
                    }
                )
        dataset_info.append(
            {
                "replicate": rep,
                "observed_sha256": data.observed_hashes,
                "truth_sha256": {
                    f: sha256(data.directory / f)
                    for f in (
                        "path_truth.json",
                        "effect_truth.json",
                        "grid.json",
                        "T.json",
                        "alpha.json",
                        "beta.json",
                    )
                },
            }
        )
        for method in METHODS:
            result, cfg = load_fit(outdir / f"replicate_{rep:03d}" / method, data, rep, method)
            req = cfg["request"]
            if method.startswith("twostage") and req.get("stage1_result_sha256") != s1hash:
                raise ValueError(f"rep {rep}: {method} uses a different Stage 1")
            if (
                method.startswith("twostage")
                and req.get("stage1_prior") != s1cfg["request"]["stage1_prior"]
            ):
                raise ValueError(f"rep {rep}: {method} has inconsistent Stage 1 hyperpriors")
            if method == "oracle_t" and req["true_t_sha256"] != sha256(data.directory / "T.json"):
                raise ValueError("Oracle truth file changed")
            settings = {
                k: req[k]
                for k in (
                    "iterations",
                    "burnin",
                    "priors",
                    "stage1_prior",
                    "version",
                    "knots",
                    "degree",
                    "source",
                    "driver_sha256",
                    "sampler_sha256",
                )
            }
            if method in request_reference and settings != request_reference[method]:
                raise ValueError(f"{method}: configuration changed across replicates")
            request_reference[method] = settings
            A, b, C = result["A"], result["b"], result["C"]
            coeff = effect_coefficients(A, b, C)
            alt = effect_coefficients(A.mean(0), b.mean(0), C.mean(0))
            for effect in truths:
                estimate, product_estimate = phi @ coeff[effect].mean(0), phi @ alt[effect]
                rise = float(np.sqrt(integral((estimate - truths[effect]) ** 2, grid)))
                alt_rise = float(np.sqrt(integral((product_estimate - truths[effect]) ** 2, grid)))
                function_rows.append(
                    {
                        "replicate": rep,
                        "K": K,
                        "n": N,
                        "method": method,
                        "effect": effect,
                        "RISE": rise,
                        "RISE_product_of_parameter_means": alt_rise,
                        "L2_difference_between_estimators": float(
                            np.sqrt(integral((estimate - product_estimate) ** 2, grid))
                        ),
                    }
                )
                curve_store[method][effect].append(estimate)
                draws = coeff[effect] @ basis_weights
                truth = float(integral(truths[effect], grid))
                point = float(draws.mean())
                integrated_rows.append(
                    {
                        "replicate": rep,
                        "method": method,
                        "target": "I" + effect,
                        "truth": truth,
                        "estimate": point,
                        "error": point - truth,
                        "posterior_sd": float(draws.std(ddof=1)),
                        "CrI025": float(np.quantile(draws, 0.025)),
                        "CrI975": float(np.quantile(draws, 0.975)),
                        "interval_interpretation": (
                            "joint_posterior" if method == "joint" else "conditional_posterior"
                        ),
                    }
                )
            for name, chain in (("A", A), ("b", b), ("C", C)):
                for j in range(K):
                    point = float(chain[:, j].mean())
                    coefficient_rows.append(
                        {
                            "replicate": rep,
                            "method": method,
                            "block": name,
                            "component": j + 1,
                            "truth": float(path_truth[name][j]),
                            "estimate": point,
                            "error": point - float(path_truth[name][j]),
                        }
                    )
            tm = data.check_t(result["T_mean"], "fitted t")
            am = finite_array(result["alpha_mean"], "alpha mean", (N, K))
            bm = finite_array(result["beta_mean"], "beta mean", (N, K))
            t_mse = np.sum(((tm - tt) * data.mask) ** 2, axis=1) / data.counts
            g_ise = integral(((am - at) @ phi.T) ** 2, grid, axis=1)
            h_ise = integral(((bm - bt) @ phi.T) ** 2, grid, axis=1)
            recovery_rows.append(
                {
                    "replicate": rep,
                    "method": method,
                    "t_RMSE_donor_balanced": float(np.sqrt(t_mse.mean())),
                    "g_IMSE_donor_balanced": float(g_ise.mean()),
                    "h_IMSE_donor_balanced": float(h_ise.mean()),
                }
            )
            d = cfg["diagnostics"]
            s1seconds = (
                s1cfg["diagnostics"]["wall_seconds"] if method.startswith("twostage") else 0.0
            )
            seconds = d.get("wall_seconds")
            timing_rows.append(
                {
                    "replicate": rep,
                    "method": method,
                    "fit_seconds": seconds,
                    "stage1_seconds": s1seconds,
                    "pipeline_seconds": seconds + s1seconds if seconds is not None else None,
                    "retained_draws": len(A),
                    "baseline_source": req["source"],
                }
            )
    report = getattr(args, "summary_dir", None) or outdir / "summary"
    report.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("function_metrics_by_replicate", function_rows),
        ("integrated_effects_by_replicate", integrated_rows),
        ("path_coefficients_by_replicate", coefficient_rows),
        ("trajectory_metrics_by_replicate", recovery_rows),
        ("stage1_hyperparameters_by_replicate", hyperparameter_rows),
        ("timing_by_replicate", timing_rows),
    ):
        write_csv(report / (name + ".csv"), rows)
    summary, paired, integrated_summary, coefficient_summary = [], [], [], []
    for method in METHODS:
        for effect in ("ME", "DE", "TE"):
            rr = [x for x in function_rows if x["method"] == method and x["effect"] == effect]
            mean, mcse = mean_mcse([x["RISE"] for x in rr])
            summary.append(
                {
                    "method": method,
                    "effect": effect,
                    "R": len(rr),
                    "mean_RISE": mean,
                    "simulation_MCSE": mcse,
                }
            )
            if method != "joint":
                baseline = {
                    x["replicate"]: x["RISE"]
                    for x in function_rows
                    if x["method"] == "joint" and x["effect"] == effect
                }
                diffs = [x["RISE"] - baseline[x["replicate"]] for x in rr]
                dm, ds = mean_mcse(diffs)
                paired.append(
                    {
                        "comparison": method + " minus joint",
                        "effect": effect,
                        "R": len(rr),
                        "mean_RISE_difference": dm,
                        "simulation_MCSE": ds,
                        "positive_means": "joint_has_lower_RISE",
                    }
                )
            ii = [
                x for x in integrated_rows if x["method"] == method and x["target"] == "I" + effect
            ]
            bias, se = mean_mcse([x["error"] for x in ii])
            rmse, rse = rmse_jackknife_mcse([x["error"] for x in ii])
            integrated_summary.append(
                {
                    "method": method,
                    "target": "I" + effect,
                    "R": len(ii),
                    "mean_estimate": float(np.mean([x["estimate"] for x in ii])),
                    "bias": bias,
                    "bias_simulation_MCSE": se,
                    "RMSE": rmse,
                    "RMSE_jackknife_MCSE": rse,
                    "mean_CrI_width": float(np.mean([x["CrI975"] - x["CrI025"] for x in ii])),
                }
            )
        for block in ("A", "b", "C"):
            rows = [x for x in coefficient_rows if x["method"] == method and x["block"] == block]
            e = np.asarray([x["error"] for x in rows]).reshape(len(reps), K)
            rmse, rse = rmse_jackknife_mcse(e)
            coefficient_summary.append(
                {
                    "method": method,
                    "block": block,
                    "R": len(reps),
                    "K": K,
                    "pooled_component_RMSE": rmse,
                    "replicate_jackknife_MCSE": rse,
                }
            )
    write_csv(report / "function_summary.csv", summary)
    write_csv(report / "paired_RISE_differences.csv", paired)
    write_csv(report / "integrated_summary.csv", integrated_summary)
    write_csv(report / "coefficient_summary.csv", coefficient_summary)
    recovery_summary = []
    for method in METHODS:
        for key in ("t_RMSE_donor_balanced", "g_IMSE_donor_balanced", "h_IMSE_donor_balanced"):
            values = [x[key] for x in recovery_rows if x["method"] == method]
            mean, se = mean_mcse(values)
            recovery_summary.append(
                {
                    "method": method,
                    "metric": key,
                    "R": len(values),
                    "mean": mean,
                    "simulation_MCSE": se,
                }
            )
    write_csv(report / "trajectory_summary.csv", recovery_summary)
    curves = []
    for method in METHODS:
        for effect in ("ME", "DE", "TE"):
            values = np.asarray(curve_store[method][effect])
            mn = values.mean(0)
            se = (
                values.std(0, ddof=1) / np.sqrt(len(reps))
                if len(reps) > 1
                else np.full(len(grid_ref), np.nan)
            )
            for j, t in enumerate(grid_ref):
                curves.append(
                    {
                        "method": method,
                        "effect": effect,
                        "t": float(t),
                        "truth": float(truth_ref[effect][j]),
                        "mean_estimate": float(mn[j]),
                        "simulation_MCSE_of_mean": float(se[j]) if np.isfinite(se[j]) else None,
                    }
                )
    write_csv(report / "effect_curve_summary.csv", curves)
    write_tables(report, summary, len(reps))
    atomic_json(
        report / "evaluation_config.json",
        {
            "K": K,
            "n": N,
            "replicates": reps,
            "R": len(reps),
            "primary_estimator": "mean_s(phi(t)^T B_s A_s); DE=phi(t)^T mean_s(C_s)",
            "integration": "trapezoid on saved grid.json, uniform dt on [0,1]",
            "MCSE": "between independent simulation replicates; distinct from MCMC MCSE",
            "coverage_computed": False,
            "datasets": dataset_info,
            "stage1_prior": request_reference["stage1"]["stage1_prior"],
        },
    )
    print(f"Summary saved: {report}")


def write_tables(report, rows, r):
    lookup = {(x["method"], x["effect"]): x for x in rows}

    def entry(method, effect):
        x = lookup[method, effect]
        a, se = 100 * x["mean_RISE"], x["simulation_MCSE"]
        return f"{a:.3f} ({100 * se:.3f})" if se is not None else f"{a:.3f} (NA)"

    md = [
        "# Two-stage comparison",
        "",
        f"K=5; n=1000; R={r}. RISE and simulation MCSE in this table are multiplied by 100.",
        "",
        "| Method | ME | DE | TE |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        md.append(
            "| "
            + LABELS[method]
            + " | "
            + " | ".join(entry(method, e) for e in ("ME", "DE", "TE"))
            + " |"
        )
    md += [
        "",
        "Primary effects average B_s A_s within posterior draws. The alternative "
        "product E(B)E(A) is retained in function_metrics_by_replicate.csv for comparison "
        "with the earlier evaluator.",
        "",
        "CSV metrics are unscaled. No frequentist coverage is calculated.",
        "R=1 has no between-replicate MCSE and is not the final comparison.",
        "Legacy joint runtime is unavailable; empty time is not zero.",
        "Each two-stage pipeline time includes Stage 1; Stage 1 was computed once.",
    ]
    md += [
        "",
        "Stage 1 uses separate hierarchical Gaussian trajectory priors with learned "
        "population means and diagonal variances, without Z or mediation regressions. "
        "Hyperparameter intervals in stage1_hyperparameters_by_replicate.csv describe "
        "each fitted dataset's posterior; they are not between-replicate intervals.",
    ]
    (report / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Effect-function recovery for joint and two-stage estimation "
        rf"with $K=5$ and $n=1000$, over {r} paired simulation replicates. "
        r"Entries are mean RISE with simulation Monte Carlo standard errors in parentheses; "
        r"both are multiplied by $10^2$. Lower is better.}",
        r"\label{tab:two_stage_comparison}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & ME & DE & TE \\",
        r"\midrule",
    ]
    for method in METHODS:
        tex.append(
            LABELS[method]
            + " & "
            + " & ".join(entry(method, e) for e in ("ME", "DE", "TE"))
            + r" \\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (report / "main_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("outputs/finite"))
    p.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT))
    p.add_argument("--summary-dir", type=Path)
    p.add_argument("--replicates", type=int, nargs="+", default=list(range(1, 11)))
    evaluate(p.parse_args())
