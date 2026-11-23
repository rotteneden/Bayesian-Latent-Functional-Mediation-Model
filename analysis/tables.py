"""Assemble manuscript tables and the finite-sample supplement plot from summaries."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def mcse(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def rmse_mcse(x):
    x = np.asarray(x, float)
    rms = float(np.sqrt(np.mean(x * x)))
    if len(x) < 2:
        return rms, np.nan
    loo = np.asarray([np.sqrt(np.mean(np.delete(x, i, axis=0) ** 2)) for i in range(len(x))])
    return rms, float(np.sqrt((len(x) - 1) * np.mean((loo - loo.mean()) ** 2)))


def save_table(df, out, stem):
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / (stem + ".csv"), index=False, float_format="%.17g")
    # Formatting is deliberately separate from numeric CSVs.
    display = df.copy()
    for col in display:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: f"{x:.4f}")
    escapes = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
    }
    tex = lambda value: "".join(escapes.get(c, c) for c in str(value))
    lines = [
        r"\begin{tabular}{" + "l" * len(display.columns) + "}",
        r"\hline",
        " & ".join(map(tex, display.columns)) + r" \\",
        r"\hline",
    ]
    lines.extend(
        " & ".join(map(tex, row)) + r" \\" for row in display.itertuples(index=False, name=None)
    )
    lines.extend([r"\hline", r"\end{tabular}"])
    (out / (stem + ".tex")).write_text("\n".join(lines) + "\n", encoding="utf-8")


def finite(root, out):
    p = pd.read_csv(root / "finite_sample_path_summary.csv")
    f = pd.read_csv(root / "finite_sample_function_summary.csv")
    detail = pd.read_csv(root / "finite_sample_path_replicate_metrics.csv")
    rows = []
    for n in sorted(p.sample_size.unique()):
        row = {"n": n}
        for block in ("A", "diag(B)", "C"):
            r = p[(p.sample_size == n) & (p.parameter_block == block)].iloc[0]
            row[block + "_RMSE_x100"] = 100 * r.rmse
            row[block + "_MCSE_x100"] = 100 * r.mcse_rmse
        for e in ("ME", "DE", "TE"):
            r = f[(f.sample_size == n) & (f.target == e + "(t)")].iloc[0]
            row[e + "_RISE_x100"] = 100 * r.mean_error
            row[e + "_MCSE_x100"] = 100 * r.mcse_mean_error
        rows.append(row)
    save_table(pd.DataFrame(rows), out, "table_1_finite")
    s1 = (
        detail.groupby(["sample_size", "parameter_block", "component"])
        .agg(bias=("error", "mean"), MSE=("squared_error", "mean"), R=("replicate", "nunique"))
        .reset_index()
    )
    s1["RMSE"] = np.sqrt(s1.pop("MSE"))
    save_table(s1, out, "table_S1_component_recovery")
    save_table(f[f.error_metric == "IMSE"], out, "table_S1_trajectory_IMSE")
    d = pd.read_csv(root / "finite_sample_function_replicate_metrics.csv")
    fig, axs = plt.subplots(2, 3, figsize=(10, 5.6), layout="constrained")
    for ax, target in zip(axs.flat, ("ME(t)", "DE(t)", "TE(t)", "g_i(t)", "h_i(t)")):
        subset = d[d.target == target]
        scale = 100 if target in ("ME(t)", "DE(t)", "TE(t)") else 10000
        for _, part in subset.groupby("replicate"):
            part = part.sort_values("sample_size")
            ax.plot(part.sample_size, scale * part.mean_error, "o-", color=".65", lw=0.8, ms=2)
        mean = subset.groupby("sample_size").mean_error.mean()
        ax.plot(mean.index, scale * mean, "o-", color="black", lw=1.8, ms=4)
        ax.set(
            xlabel="Donors", ylabel=("RISE × 100" if scale == 100 else "IMSE × 10000"), title=target
        )
        ax.set_xticks(sorted(subset.sample_size.unique()))
        ax.spines[["top", "right"]].set_visible(False)
    axs[-1, -1].axis("off")
    fig.savefig(out / "figure_S2_finite.pdf")
    fig.savefig(out / "figure_S2_finite.png", dpi=200)
    plt.close(fig)


def basis(root, cv, out):
    summary = pd.read_csv(root / "basis_robustness_summary.csv")
    rep = pd.read_csv(root / "basis_robustness_replicate_metrics.csv")
    cv_detail = pd.read_csv(cv / "cv_scores_by_replicate.csv")
    cv_detail = cv_detail[(cv_detail.scope == "all") & (cv_detail.weighting == "donor_balanced")]
    fit_pairs = set(zip(rep["replicate"], rep["K"]))
    cv_pairs = set(zip(cv_detail["replicate"], cv_detail["K"]))
    if fit_pairs != cv_pairs:
        raise ValueError(
            "Basis recovery and conditional CV must use the same (replicate, K) pairs."
        )
    score = pd.read_csv(cv / "overall_cv_summary.csv")
    score = score[(score.scope == "all") & (score.weighting == "donor_balanced")]
    if set(score.K) != set(summary.K):
        raise ValueError("Missing basis CV candidates.")
    merged = summary.merge(score, on="K", validate="one_to_one", suffixes=("_fit", "_cv"))
    if not (merged.n_replicates_fit == merged.n_replicates_cv).all():
        raise ValueError("Different CV/recovery replicate counts.")
    rows = []
    for _, r in merged.iterrows():
        row = {"K": int(r.K)}
        for e in ("ME", "DE", "TE"):
            row[e + "_RISE_x100"] = 100 * r["RISE_" + e]
            row[e + "_MCSE_x100"] = 100 * r["MCSE_RISE_" + e]
        row.update(conditional_CVLPD=r.mean_lpd_xy, CVLPD_MCSE=r.simulation_mcse_lpd_xy)
        rows.append(row)
    save_table(pd.DataFrame(rows), out, "table_2_basis")
    save_table(
        summary[["K", "IMSE_g", "MCSE_IMSE_g", "IMSE_h", "MCSE_IMSE_h", "n_replicates"]],
        out,
        "table_S2_trajectory_IMSE",
    )
    rows = []
    for K, g in rep.groupby("K"):
        for e in ("IME", "IDE", "ITE"):
            error = g[e + "_error"].to_numpy()
            rms, rse = rmse_mcse(error)
            rows.append(
                dict(
                    K=K,
                    effect=e,
                    R=len(g),
                    mean_estimate=g[e + "_estimate"].mean(),
                    truth=g[e + "_truth"].mean(),
                    bias=error.mean(),
                    bias_MCSE=mcse(error),
                    RMSE=rms,
                    RMSE_MCSE=rse,
                )
            )
    save_table(pd.DataFrame(rows), out, "table_S3_integrated_recovery")
    scores = pd.read_csv(cv / "cv_scores_by_replicate.csv")
    scores = scores[(scores.scope == "all") & (scores.weighting == "donor_balanced")]
    if scores.duplicated(["replicate", "K"]).any():
        raise ValueError("Duplicate per-replicate scores.")
    if set(scores.replicate) != set(rep.replicate):
        raise ValueError("Different replicate identities in CV and recovery.")
    selected = (
        scores.sort_values(["replicate", "lpd_xy", "K"], ascending=[True, False, True])
        .groupby("replicate")
        .head(1)
    )
    selected = selected[["replicate", "K", "lpd_xy"]].merge(
        rep, on=["replicate", "K"], validate="one_to_one"
    )
    save_table(selected, out, "selected_K_full_model_recovery")
    wide = scores.pivot(index="replicate", columns="K", values="lpd_xy")
    if 5 in wide and 6 in wide:
        d = (wide[5] - wide[6]).dropna()
        save_table(
            pd.DataFrame(
                [{"contrast": "K5 minus K6", "R": len(d), "mean": d.mean(), "MCSE": mcse(d)}]
            ),
            out,
            "paired_CVLPD_K5_minus_K6",
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", choices=["finite", "basis"])
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--cv", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.experiment == "finite":
        finite(a.summary, a.output)
    else:
        if not a.cv:
            p.error("--cv is required for basis Table 2.")
        basis(a.summary, a.cv, a.output)
