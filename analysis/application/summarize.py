"""Summarize the selected OneK1K fit, diagnostics and conditional residual-correlation analysis."""

from __future__ import annotations

import argparse
from array import array
import gc
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import shutil
import sys
import time

import numpy as np
import pandas as pd
from blafm.mcmc import cubic_basis
from scipy.stats import t as student_t, geninvgauss
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import ijson
    from ijson.common import ObjectBuilder
    from tqdm import tqdm
except ImportError as exc:
    raise SystemExit(
        "Install dependencies: python -m pip install numpy scipy pandas matplotlib ijson tqdm"
    ) from exc

VERSION = "1.2.0-paper-reproduction"
METHOD = "application_scheme2_fixed_own_K_T_pred"
K_PRIMARY = 7
RHOS = [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]

plt.style.use("default")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 600,
    }
)

# Gray shades + distinct linetypes make rho curves distinguishable in print.
RHO_STYLES = {
    -0.5: {"color": "0.22", "linestyle": "-", "linewidth": 1.65},
    -0.3: {"color": "0.42", "linestyle": (0, (7, 2)), "linewidth": 1.45},
    -0.1: {"color": "0.63", "linestyle": (0, (3, 1.5)), "linewidth": 1.40},
    0.0: {"color": "0.08", "linestyle": "--", "linewidth": 2.35},
    0.1: {"color": "0.52", "linestyle": "-.", "linewidth": 1.40},
    0.3: {"color": "0.72", "linestyle": ":", "linewidth": 1.85},
    0.5: {"color": "0.34", "linestyle": (0, (7, 2, 1, 2)), "linewidth": 1.40},
}
K_STYLES = {
    4: {"color": "0.62", "linestyle": "--", "linewidth": 1.25, "zorder": 1},
    5: {"color": "0.35", "linestyle": (0, (8, 2)), "linewidth": 1.55, "zorder": 2},
    6: {"color": "0.48", "linestyle": (0, (5, 2, 1, 2)), "linewidth": 1.35, "zorder": 2},
    7: {"color": "0.08", "linestyle": "-", "linewidth": 2.30, "zorder": 6},
    8: {"color": "0.28", "linestyle": "-.", "linewidth": 1.60, "zorder": 3},
    9: {"color": "0.80", "linestyle": ":", "linewidth": 1.65, "zorder": 1},
}


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.17g")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def vector_hash(rows):
    h = hashlib.sha256()
    for row in rows:
        a = np.asarray(row, dtype="<f8")
        h.update(np.asarray([len(a)], dtype="<i8").tobytes())
        h.update(a.tobytes())
    return h.hexdigest()


def resolve_input(value, root):
    """Honor exact paths, then relocate Windows application paths by basename."""
    p = Path(value)
    candidates = [p if p.is_absolute() else root / p]
    w = PureWindowsPath(str(value))
    if w.is_absolute() or p.is_absolute():
        parts = w.parts if w.is_absolute() else p.parts
        if "application" in parts:
            i = max(j for j, x in enumerate(parts) if x == "application")
            candidates.append(root.joinpath(*parts[i + 1 :]))
        if len(parts) >= 2:
            candidates.append(root / parts[-2] / parts[-1])
        candidates.append(root / parts[-1])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Input not found: %s. Set --root to your application directory." % value
    )


def integer(x, label):
    f = float(x)
    if not math.isfinite(f) or not f.is_integer():
        raise ValueError("%s must be an integer" % label)
    return int(f)


def load_selection(args):
    root = Path(args.root).resolve()
    cfg_path = resolve_input(args.config, root)
    cfg = read_json(cfg_path)
    cv_dir = Path(args.cv_dir or cfg.get("output_dir", "application_cv_scheme2"))
    if not cv_dir.is_absolute():
        cv_dir = root / cv_dir
    if not cv_dir.is_dir():
        cv_dir = root / PureWindowsPath(str(args.cv_dir or cfg["output_dir"])).name
    paths = {
        name: cv_dir / name for name in ("selection.json", "run_config.json", "cv_summary_by_K.csv")
    }
    sel, run = read_json(paths["selection.json"]), read_json(paths["run_config.json"])
    if sel.get("method") != METHOD or run.get("method") != METHOD:
        raise ValueError("These are not Scheme 2 application results.")
    expected = ("all", "donor_balanced", "lpd_xy")
    if tuple(sel.get(k) for k in ("scope", "weighting", "component")) != expected:
        raise ValueError(
            "Selection rule differs from the pre-specified all/donor_balanced/lpd_xy rule."
        )
    if sel.get("complete") is not True or sel.get("missing_folds"):
        raise ValueError("CV selection is incomplete.")
    candidates = list(range(4, 10))
    for v in (sel.get("candidate_Ks"), sel.get("complete_Ks"), cfg.get("candidate_Ks")):
        if sorted(v or []) != candidates:
            raise ValueError("Expected completed candidates K=4,...,9.")
    table = pd.read_csv(paths["cv_summary_by_K.csv"], float_precision="round_trip")
    target = table[(table.scope == "all") & (table.weighting == "donor_balanced")].copy()
    if sorted(target.K.tolist()) != candidates or target.K.duplicated().any():
        raise ValueError("CV summary lacks unique K=4,...,9 selection rows.")
    if not np.isfinite(target.lpd_xy).all() or not (target.n_folds == 5).all():
        raise ValueError("CV score is invalid or some folds are missing.")
    best = target.lpd_xy.max()
    winner = int(target.loc[target.lpd_xy == best, "K"].min())
    if winner != K_PRIMARY or sel.get("selected_K") != K_PRIMARY:
        raise ValueError("Expected Scheme 2 to select K=7; selection and CSV must agree.")
    if float(sel["best_lpd_xy"]) != float(best):
        raise ValueError("selection.json and score CSV disagree at full precision.")
    rc = run["configuration"]
    if cfg.get("z_offset") != rc.get("z_offset"):
        raise ValueError("Configuration z_offset differs from completed CV.")
    if cfg.get("hyperparameters") != rc.get("hyperparameters"):
        raise ValueError("Prior hyperparameters differ from the completed CV configuration.")
    files = {
        k: resolve_input(cfg["posterior_files"][str(k)], root)
        for k in ([7] if args.skip_basis else candidates)
    }
    selected_path = resolve_input(sel["full_latent_T_posterior_file"], root)
    if files[7] != selected_path:
        raise ValueError(
            "K=7 input differs from the full latent-time posterior recorded by selection.json."
        )
    for k, p in files.items():
        if p != resolve_input(rc["posterior_files"][str(k)], root):
            raise ValueError("K=%d posterior differs from the completed CV configuration." % k)
    data_path = resolve_input(cfg["data_json"], root)
    if file_hash(data_path) != run["data_sha256"]:
        raise ValueError("Data SHA256 differs from the completed Scheme 2 run.")
    target["rank"] = target.lpd_xy.rank(method="min", ascending=False).astype(int)
    target["difference_from_K7"] = target.lpd_xy - best
    return root, cfg, run, sel, table, target, files, data_path, paths


def read_posterior(path, K, args, primary=False):
    """One streaming pass. Retain diagonals, never the huge unused cell chains."""
    chain_keys = {"A", "B", "C", "alpha0", "beta0", "Sigma", "Lambda", "tau_a", "tau_d"}
    matrix_keys = {"B", "Sigma", "Lambda"}
    scalar_keys = {"tau_a", "tau_d"}
    wanted = {
        "K",
        "init",
        "burnin",
        "burin",
        "n_itrs",
        "degree",
        "rho_fixed",
        "rho",
        "fold",
        "conditioning",
        "T_pred",
        "alpha_pred",
        "beta_pred",
        "aux_saved_itr",
        "alpha0_HAT",
        "A_HAT",
        "beta0_HAT",
        "B_HAT",
        "C_HAT",
        "Sigma_HAT",
        "Lambda_HAT",
    }
    if not primary:
        chain_keys = {"A", "B", "C"}
        wanted -= {"alpha_pred", "beta_pred", "Sigma_HAT", "Lambda_HAT"}
    buffers = {k: array("d") for k in chain_keys}
    counts = {k: 0 for k in chain_keys}
    meta, builder, target, depth, is_draw = {}, None, None, 0, False

    def accept(key, value, draw):
        if not draw:
            meta[key] = value
            return
        a = np.asarray(value, dtype=float)
        if key in matrix_keys:
            if a.shape != (K, K) or not np.allclose(a, np.diag(np.diag(a)), rtol=0, atol=1e-12):
                raise ValueError("%s: %s must be a diagonal %d by %d matrix." % (path, key, K, K))
            a = np.diag(a)
        else:
            if key not in scalar_keys and a.shape not in ((K,), (1, K), (K, 1)):
                raise ValueError("%s: malformed %s draw shape %s" % (path, key, a.shape))
            a = a.reshape(-1)
        if a.size != (1 if key in scalar_keys else K) or not np.isfinite(a).all():
            raise ValueError("%s: invalid %s draw" % (path, key))
        buffers[key].extend(a)
        counts[key] += 1

    print("Reading full posterior K=%d: %s" % (K, path), flush=True)
    with Path(path).open("rb") as f:
        for prefix, event, value in ijson.parse(f, use_float=True):
            if builder is not None:
                builder.event(event, value)
                depth += int(event in ("start_map", "start_array")) - int(
                    event in ("end_map", "end_array")
                )
                if depth == 0:
                    accept(target, builder.value, is_draw)
                    builder = None
            elif prefix in wanted and event not in ("map_key", "end_map", "end_array"):
                if event in ("start_map", "start_array"):
                    builder, target, depth, is_draw = ObjectBuilder(), prefix, 1, False
                    builder.event(event, value)
                else:
                    accept(prefix, value, False)
            elif prefix.endswith(".item") and prefix[:-5] in chain_keys:
                key = prefix[:-5]
                if event in ("start_map", "start_array"):
                    builder, target, depth, is_draw = ObjectBuilder(), key, 1, True
                    builder.event(event, value)
                elif event not in ("map_key", "end_map", "end_array"):
                    accept(key, value, True)
    for obj in (meta, meta.get("init", {})):
        if "K" in obj and integer(obj["K"], "K") != K:
            raise ValueError("Full posterior K metadata conflicts with coefficient dimension.")
        if "degree" in obj and obj["degree"] != 3:
            raise ValueError("Expected cubic B-splines.")
        if any(float(obj.get(r, 0)) != 0 for r in ("rho", "rho_fixed")):
            raise ValueError("Primary/basis posterior must be the rho=0 full fit.")
    if "fold" in meta or "conditioning" in meta or "T_pred" not in meta:
        raise ValueError("Provide the FULL latent-time fit, not a CV or sensitivity fit.")
    for key in ("A_HAT", "C_HAT", "B_HAT"):
        if key in meta:
            a = np.asarray(meta[key])
            if (key == "B_HAT" and a.shape != (K, K)) or (key != "B_HAT" and a.size != K):
                raise ValueError("K=%d disagrees with %s." % (K, key))
    burn = (
        args.full_burnin if args.full_burnin is not None else meta.get("burnin", meta.get("burin"))
    )
    nit = args.full_iterations if args.full_iterations is not None else meta.get("n_itrs")
    if burn is None or nit is None:
        raise ValueError(
            "Missing burnin/n_itrs; supply --full-burnin and --full-iterations explicitly."
        )
    burn, nit = integer(burn, "burnin"), integer(nit, "n_itrs")
    if not 0 <= burn < nit or nit - burn < 4:
        raise ValueError("Require at least four retained draws.")
    if "burnin" in meta and burn != int(meta["burnin"]):
        raise ValueError("--full-burnin conflicts with stored metadata.")
    if "n_itrs" in meta and nit != int(meta["n_itrs"]):
        raise ValueError("--full-iterations conflicts with stored metadata.")
    n = counts["A"]
    start = burn + 1 if n == nit + 1 else burn if n == nit else None
    if start is None:
        raise ValueError(
            "Chain length %d is neither n_itrs nor n_itrs+1; refusing to guess thinning." % n
        )
    full = {}
    required = ("A", "B", "C", "alpha0", "beta0") if primary else ("A", "B", "C")
    for key in required:
        if counts[key] != n:
            raise ValueError("A/B/C/intercept chain lengths differ or a chain is missing.")
        full[key] = np.frombuffer(buffers[key], dtype=float).reshape(n, K)[start:].copy()
    if primary:
        aux_itr = np.asarray(meta.get("aux_saved_itr", []), dtype=int)
        for key in ("Sigma", "Lambda", "tau_a", "tau_d"):
            count = counts[key]
            if not count:
                continue
            a = np.frombuffer(buffers[key], dtype=float).reshape(count, -1)
            if len(aux_itr) == count:
                if (
                    np.any(aux_itr <= burn)
                    or np.any(aux_itr > nit)
                    or np.any(np.diff(aux_itr) <= 0)
                ):
                    raise ValueError("Invalid auxiliary draw iteration indices.")
                kept = a
            elif count == n:
                kept = a[start:]
            else:
                raise ValueError("Cannot identify post-burn-in %s auxiliary draws." % key)
            if np.any(kept <= 0):
                raise ValueError("Non-positive posterior variance/scale.")
            full[key] = kept.copy()
    info = {
        "K": K,
        "path": str(path),
        "sha256": file_hash(path),
        "burnin": burn,
        "n_itrs": nit,
        "saved_chain_length": n,
        "start_index": start,
        "n_retained": nit - burn,
        "has_initial_state": n == nit + 1,
        "basis_degree": 3,
    }
    print(
        "  K=%d verified by coefficient dimensions; retained draws=%d, start index=%d."
        % (K, nit - burn, start),
        flush=True,
    )
    return full, meta, info


def load_data(path, cfg):
    raw = read_json(path)
    keys = cfg["data_keys"]
    x = [np.asarray(a, dtype=float).reshape(-1) for a in raw[keys["x"]]]
    y = [np.asarray(a, dtype=float).reshape(-1) for a in raw[keys["y"]]]
    state = [np.asarray(a, dtype=float).reshape(-1) for a in raw[keys["cell_type"]]]
    z = np.asarray(raw[keys["z"]], dtype=float).reshape(-1) + float(cfg["z_offset"])
    if len(x) != len(y) or len(x) != len(state) or len(x) != len(z):
        raise ValueError("Data donor dimensions disagree.")
    if not np.isin(z, [0.0, 1.0, 2.0]).all():
        raise ValueError("Expected genotype dosage 0/1/2 after the recorded z_offset.")
    for a, b, s in zip(x, y, state):
        if (
            len(a) == 0
            or len(a) != len(b)
            or len(a) != len(s)
            or not np.isfinite(a).all()
            or not np.isfinite(b).all()
            or not np.isin(s, [0, 1]).all()
        ):
            raise ValueError("Invalid expression/state rows or inconsistent cell counts.")
    counts = np.asarray([len(a) for a in x])
    means = np.array([[a.mean(), b.mean()] for a, b in zip(x, y)])
    sample = pd.DataFrame(
        [
            dict(
                n_donors=len(z),
                n_cells=int(counts.sum()),
                n_early_cells=sum(int((s == 0).sum()) for s in state),
                n_late_cells=sum(int((s == 1).sum()) for s in state),
                min_cells_per_donor=int(counts.min()),
                median_cells_per_donor=float(np.median(counts)),
                max_cells_per_donor=int(counts.max()),
                n_Z0=int((z == 0).sum()),
                n_Z1=int((z == 1).sum()),
                n_Z2=int((z == 2).sum()),
                z_offset=float(cfg["z_offset"]),
            )
        ]
    )
    X = np.column_stack([np.ones(len(z)), means[:, 0], z])
    if np.linalg.matrix_rank(X) != 3 or len(z) <= 3:
        raise ValueError("Donor-average regression design is singular or too small.")
    coef = np.linalg.lstsq(X, means[:, 1], rcond=None)[0]
    resid = means[:, 1] - X @ coef
    df = len(z) - 3
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * float(resid @ resid) / df)
    crit = student_t.ppf(0.975, df)
    bench = pd.DataFrame(
        [
            dict(
                coefficient="donor_mean_RPS26",
                estimate=coef[1],
                se=se[1],
                **{"ci_2.5": coef[1] - crit * se[1], "ci_97.5": coef[1] + crit * se[1]},
                t=coef[1] / se[1],
                df=df,
                p_value=2 * student_t.sf(abs(coef[1] / se[1]), df),
                n_donors=len(z),
                model="mean(BAX) ~ 1 + mean(RPS26) + genotype_dosage",
            )
        ]
    )
    return z, counts, state, sample, bench


def verify_times(meta, K, counts, state, run):
    source = run["fixed_T_sources"][str(K)]
    rows = meta["T_pred"]
    if len(rows) != len(counts):
        raise ValueError("T_pred donor count mismatch.")
    trimmed = []
    for row, n, s in zip(rows, counts, state):
        t = np.asarray(row, dtype=float).reshape(-1)
        if len(t) not in (n, int(counts.max())):
            raise ValueError("T_pred padding/cell counts mismatch.")
        t = t[:n]
        if not np.isfinite(t).all() or np.any(t < 0.5 * s) or np.any(t > 0.5 * s + 0.5):
            raise ValueError("T_pred is non-finite or outside coarse-state support.")
        trimmed.append(t)
    if vector_hash(trimmed) != source["trimmed_times_sha256"]:
        raise ValueError("K=%d T_pred differs from the one used for Scheme 2." % K)
    # These application data use explicitly declared identical row order.
    # A matching hash certifies the same arrays, not independently verified IDs.
    return source.get("alignment", "not_recorded")


def basis(t, K):
    return cubic_basis(K)(np.asarray(t))


def basis_moments(K):
    """4-point Gauss-Legendre on each knot interval: exact degree-6 products."""
    breaks = np.linspace(0.0, 1.0, K - 2)
    g, w = np.polynomial.legendre.leggauss(4)
    t = np.concatenate([(a + b) / 2 + (b - a) * g / 2 for a, b in zip(breaks[:-1], breaks[1:])])
    weights = np.concatenate([(b - a) * w / 2 for a, b in zip(breaks[:-1], breaks[1:])])
    phi = basis(t, K)
    return phi.T @ weights, phi.T @ (weights[:, None] * phi)


def effect_coefficients(full):
    me = full["A"] * full["B"]  # B contains its saved diagonal, matching draw indices.
    return {"ME": me, "DE": full["C"], "TE": me + full["C"]}


def summarize(x):
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite posterior functional.")
    lo, hi = np.quantile(x, [0.025, 0.975])
    mean = float(x.mean())
    return dict(
        posterior_mean=mean,
        posterior_sd=float(x.std(ddof=1)),
        **{"cri_2.5": float(lo), "cri_97.5": float(hi)},
        p_gt_0=float((x > 0).mean()),
        p_lt_0=float((x < 0).mean()),
        sign_direction=">0" if mean >= 0 else "<0",
        sign_probability=float((x > 0).mean() if mean >= 0 else (x < 0).mean()),
        n_draws=len(x),
    )


def curve_summary(coef, design, block=24):
    out = {k: np.empty(len(design)) for k in ("mean", "sd", "low", "high")}
    for start in range(0, len(design), block):
        sl = slice(start, start + block)
        values = coef @ design[sl].T
        out["mean"][sl] = values.mean(axis=0)
        out["sd"][sl] = values.std(axis=0, ddof=1)
        out["low"][sl], out["high"][sl] = np.quantile(values, [0.025, 0.975], axis=0)
    return out


def shape_summary(t, s, label):
    mean, low, high = s["mean"], s["low"], s["high"]
    return dict(
        quantity=label,
        fraction_mean_negative=float((mean < 0).mean()),
        fraction_mean_positive=float((mean > 0).mean()),
        fraction_CrI_below_zero=float((high < 0).mean()),
        fraction_CrI_above_zero=float((low > 0).mean()),
        minimum_mean=float(mean.min()),
        t_at_minimum=float(t[mean.argmin()]),
        maximum_mean=float(mean.max()),
        t_at_maximum=float(t[mean.argmax()]),
    )


def integrated(full, K):
    m, _ = basis_moments(K)
    chains = {"I" + name: coef @ m for name, coef in effect_coefficients(full).items()}
    return pd.DataFrame([dict(effect=name, **summarize(x)) for name, x in chains.items()]), chains


def ess_diagnostic(name, x):
    """Traditional ACF ESS, positive paired-lag truncation (1+2,3+4,...)."""
    x = np.asarray(x, dtype=float).reshape(-1)
    n = len(x)
    sd = float(x.std(ddof=1))
    centered = x - x.mean()
    if np.all(centered == 0):
        ess = mcse = ratio = float("nan")
        status = "constant_chain_check_sampler"
    else:
        size = 1 << (2 * n - 1).bit_length()
        f = np.fft.rfft(centered, n=size)
        acov = np.fft.irfft(f * np.conjugate(f), n=size)[:n] / np.arange(n, 0, -1)
        acf = acov / acov[0]
        total = 0.0
        for j in range(1, n - 1, 2):
            pair = acf[j] + acf[j + 1]
            if pair <= 0:
                break
            total += pair
        ess = float(n / max(1.0, 1.0 + 2.0 * total))
        mcse = sd / math.sqrt(ess)
        ratio = mcse / sd
        status = "inspect_trace" if ess < 100 else "not_a_convergence_certificate"
    return dict(
        quantity=name,
        posterior_mean=float(x.mean()),
        posterior_sd=sd,
        ESS=ess,
        MCSE=mcse,
        MCSE_over_SD=ratio,
        n_draws=n,
        status=status,
    )


def diagnostics(
    full, ichains, out, stem="K7", iteration_start=1, display_max_points=12000, smooth_window=151
):
    """ESS/MCSE use every post-burn-in draw; figure smoothing is display only.

    Trace plots cover integrated quantities
    form four stacked panels; A/diag(B)/C form a row of K coefficient panels.
    """
    items = {}
    for key in ("alpha0", "A", "beta0", "B", "C"):
        for k in range(full[key].shape[1]):
            items["%s[%d]" % ("diagB" if key == "B" else key, k + 1)] = full[key][:, k]
    items.update(ichains)
    items["integrated_gamma"] = full["B"].sum(axis=1)
    diag = pd.DataFrame([ess_diagnostic(k, v) for k, v in items.items()])
    csv(diag, out / ("mcmc_diagnostics_" + stem + ".csv"))

    # The visual horizontal coordinate follows the AoAS reference script:
    # zero-indexed retained post-burn-in draw, not the original MCMC iteration.
    trace_entries = [(name, ichains[name]) for name in ("IME", "IDE", "ITE")]
    trace_entries.append((r"$\int \gamma$", items["integrated_gamma"]))
    fig, axes = plt.subplots(4, 1, figsize=(9.2, 7.8), sharex=True)
    for j, (ax, (label, chain)) in enumerate(zip(axes, trace_entries)):
        plot_trace_line(ax, chain, display_max_points, smooth_window)
        ax.set_ylabel(label)
        apply_trace_axis_style(ax)
        panel_label(ax, "(%s)" % chr(ord("a") + j), x=-0.10, y=1.02)
    axes[-1].set_xlabel("Retained post-burn-in draw index")
    fig.subplots_adjust(left=0.10, right=0.985, top=0.98, bottom=0.09, hspace=0.18)
    save_figure(
        fig,
        out / ("trace_integrated_functionals_" + stem + ".pdf"),
        out / ("trace_integrated_functionals_" + stem + ".png"),
    )

    for family, family_name, labels in (
        ("A", r"$A_k$", [r"$A_%d$" % (k + 1) for k in range(full["A"].shape[1])]),
        ("diagB", r"$B_{kk}$", [r"$B_{%d%d}$" % (k + 1, k + 1) for k in range(full["B"].shape[1])]),
        ("C", r"$C_k$", [r"$C_%d$" % (k + 1) for k in range(full["C"].shape[1])]),
    ):
        values = full["B"] if family == "diagB" else full[family]
        plot_trace_family(
            values,
            family_name,
            labels,
            out / ("trace_" + family + "_" + stem + ".pdf"),
            out / ("trace_" + family + "_" + stem + ".png"),
            display_max_points,
            smooth_window,
        )

    # As in the reference script, also supply one consolidated 3 x K trace figure.
    plot_combined_abc_traces(
        full,
        out / ("trace_ABC_" + stem + ".pdf"),
        out / ("trace_ABC_" + stem + ".png"),
        display_max_points,
        smooth_window,
    )
    return diag


def write_effect_table(df, path):
    from analysis.tables import save_table

    display = df[["effect", "posterior_mean", "posterior_sd", "cri_2.5", "cri_97.5", "ESS", "MCSE"]]
    save_table(display, path.parent, path.stem)


def primary_outputs(full, info, args, out):
    main, supp = out / "main", out / "supplement"
    t = np.linspace(0, 1, args.grid_size)
    phi = basis(t, 7)
    effects = effect_coefficients(full)
    edf, ic = integrated(full, 7)
    pd_draws = pd.DataFrame(dict(iteration=np.arange(info["burnin"] + 1, info["n_itrs"] + 1), **ic))
    csv(pd_draws, supp / "integrated_effect_draws_K7.csv")
    es = {name: curve_summary(c, phi) for name, c in effects.items()}
    frame = {"pseudotime": t}
    for name, s in es.items():
        for stat, val in s.items():
            frame[name + "_" + stat] = val
    csv(pd.DataFrame(frame), main / "primary_effect_curves_K7.csv")
    shape = pd.DataFrame([shape_summary(t, s, name) for name, s in es.items()])
    csv(shape, main / "primary_effect_shape_summary_K7.csv")
    ts = {"RPS26": {}, "BAX": {}}
    tf = {"pseudotime": t}
    for z in (0, 1, 2):
        a = full["alpha0"] + z * full["A"]
        b = full["beta0"] + z * full["C"] + full["B"] * a
        for gene, c in [("RPS26", a), ("BAX", b)]:
            s = curve_summary(c, phi)
            ts[gene][z] = s
            for stat, val in s.items():
                tf["%s_Z%d_%s" % (gene, z, stat)] = val
    csv(pd.DataFrame(tf), main / "primary_mean_trajectories_K7.csv")
    contrast = {"pseudotime": t}
    for label, c in [("RPS26_Z2_minus_Z0", 2 * full["A"]), ("BAX_Z2_minus_Z0", 2 * effects["TE"])]:
        for stat, val in curve_summary(c, phi).items():
            contrast[label + "_" + stat] = val
    csv(pd.DataFrame(contrast), main / "trajectory_genotype_contrasts_K7.csv")
    plot_primary_figure(
        t, es, ts, main / "app_primary_results_K7.pdf", main / "app_primary_results_K7.png"
    )
    gint = pd.DataFrame([dict(quantity="integrated_gamma", **summarize(full["B"].sum(axis=1)))])
    csv(gint, main / "integrated_gamma_K7.csv")

    diag = diagnostics(
        full,
        ic,
        supp,
        iteration_start=info["burnin"] + 1,
        display_max_points=args.trace_display_max_points,
        smooth_window=args.trace_smooth_window,
    )
    precision = diag[["quantity", "ESS", "MCSE"]]
    edf = edf.merge(precision, left_on="effect", right_on="quantity", validate="one_to_one").drop(
        columns="quantity"
    )
    csv(edf, main / "integrated_effects_K7.csv")
    write_effect_table(edf, main / "table_6_integrated_effects_K7.tex")
    table_s10 = pd.concat(
        [edf.rename(columns={"effect": "quantity"}).drop(columns=["ESS", "MCSE"]), gint],
        ignore_index=True,
    )
    table_s10 = table_s10.merge(precision, on="quantity", validate="one_to_one")
    from analysis.tables import save_table

    save_table(table_s10, supp, "table_S10_integrated_diagnostics_K7")
    return edf, gint, shape, diag


def selection_outputs(table, target, sel, out):
    csv(table, out / "supplement" / "scheme2_cv_all_scores.csv")
    csv(target.sort_values("K"), out / "main" / "scheme2_K_selection.csv")
    q = target.sort_values("K")
    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    ax.plot(
        q.K,
        q.lpd_xy,
        color="0.08",
        lw=1.65,
        marker="o",
        ms=4.0,
        markerfacecolor="white",
        markeredgecolor="0.08",
        markeredgewidth=0.9,
    )
    ax.plot(
        [7],
        [sel["best_lpd_xy"]],
        linestyle="None",
        marker="o",
        ms=8,
        markerfacecolor="white",
        markeredgecolor="0.08",
        markeredgewidth=1.5,
        zorder=5,
    )
    ax.set(
        xlabel=r"Basis dimension $K$",
        ylabel="Donor-balanced conditional plug-in CVLPD",
        xticks=list(range(4, 10)),
    )
    ax.tick_params(length=3.5, width=0.8)
    fig.subplots_adjust(left=0.18, right=0.97, top=0.95, bottom=0.17)
    save_figure(
        fig, out / "main" / "scheme2_K_selection.pdf", out / "main" / "scheme2_K_selection.png"
    )
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Scheme 2 conditional plug-in selection scores. Each candidate uses its own full-fit $\widehat T_K$. The largest donor-balanced joint score selects $K=7$.}",
        r"\label{tab:app_k_selection}",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"$K$ & LPD$_X$ & LPD$_{Y\mid X}$ & Joint LPD & Rank \\",
        r"\midrule",
    ]
    for _, r in q.iterrows():
        lines.append(
            "%d & %.4f & %.4f & %.4f & %d \\" % (r.K, r.lpd_x, r.lpd_y_given_x, r.lpd_xy, r["rank"])
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "main" / "scheme2_K_selection.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def basis_outputs(loaded, paths, args, counts, state, run, out, manifest):
    t = np.linspace(0, 1, args.grid_size)
    ints, points = [], []
    for k, path in sorted(paths.items()):
        if k == 7:
            full = loaded
        else:
            full, meta, info = read_posterior(path, k, args)
            info["time_alignment"] = verify_times(meta, k, counts, state, run)
            manifest["basis_posteriors"][str(k)] = info
            del meta
        df, _ = integrated(full, k)
        df.insert(0, "K", k)
        df["fit_file"] = str(path)
        ints.append(df)
        for effect, c in effect_coefficients(full).items():
            points.append(
                pd.DataFrame(
                    dict(
                        K=k,
                        pseudotime=t,
                        effect=effect,
                        posterior_mean=basis(t, k) @ c.mean(axis=0),
                    )
                )
            )
        if k != 7:
            del full
        gc.collect()
    idf = pd.concat(ints, ignore_index=True)
    pdf = pd.concat(points, ignore_index=True)
    supp = out / "supplement"
    csv(idf, supp / "basis_sensitivity_integrated_effects_K4_K9.csv")
    csv(pdf, supp / "basis_sensitivity_pointwise_means_K4_K9.csv")

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7), sharex=True)
    for j, (ax, effect, title) in enumerate(
        zip(axes, ("ME", "DE", "TE"), ("Mediated effect", "Direct effect", "Total effect"))
    ):
        # Draw the selected K=7 last; ensure it remains visible in gray-scale print.
        for k in (4, 5, 6, 8, 9, 7):
            q = pdf[(pdf.K == k) & (pdf.effect == effect)]
            ax.plot(q.pseudotime, q.posterior_mean, label=r"$K=%d$" % k, **K_STYLES[k])
        ax.axhline(0, color="0.76", linestyle="--", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Pseudotime")
        ax.set_ylabel(r"$%s(t)$" % effect)
        apply_curve_axis_style(ax)
        panel_label(ax, "(%s)" % chr(ord("a") + j))
    handles, labels = axes[0].get_legend_handles_labels()
    first = dict(zip(labels, handles))
    order = [4, 5, 6, 7, 8, 9]
    fig.legend(
        [first[r"$K=%d$" % k] for k in order],
        [r"$K=%d$" % k + (" (selected)" if k == 7 else "") for k in order],
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        handlelength=3.0,
        columnspacing=1.25,
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.27, wspace=0.34)
    save_figure(
        fig,
        supp / "application_basis_sensitivity_K4_K9.pdf",
        supp / "application_basis_sensitivity_K4_K9.png",
    )


def prepare_output(path, overwrite, inputs):
    path = path.resolve()
    marker = path / "regeneration_manifest.json"
    for src in inputs:
        if src.resolve() == path or path in src.resolve().parents:
            raise ValueError(
                "Output directory contains an input; use a separate results directory."
            )
    if path.exists() and any(path.iterdir()):
        if not marker.is_file() or read_json(marker).get("generator") not in (
            "summarize_application",
            "regenerate_application_K7_results",
        ):
            raise ValueError("Output is not owned by this script; choose an empty output folder.")
        if not overwrite:
            raise FileExistsError(
                "Output exists. Add --overwrite to regenerate tables/figures and reuse compatible rho caches."
            )
        # Remove only manifest-tracked products to avoid stale optional results.
        old = read_json(marker)
        for rel in old.get("generated_files", []):
            p = path / rel
            if p.resolve().is_relative_to(path) and p.is_file() and p.parts[-2] != "rho_fits":
                p.unlink()
    for folder in ("main", "supplement", "provenance", "rho_fits"):
        (path / folder).mkdir(parents=True, exist_ok=True)
    return path


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root", default=str(Path(__file__).resolve().parent))
    p.add_argument("--config", default="settings.json")
    p.add_argument("--cv-dir", default=None)
    p.add_argument("--output", default="../../outputs/application/summary")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-basis", action="store_true")
    p.add_argument("--rho-mode", choices=["refit", "skip"], default="refit")
    p.add_argument("--rho-iterations", type=int, default=100000)
    p.add_argument("--rho-burnin", type=int, default=50000)
    p.add_argument("--rho-seed", type=int, default=1102)
    p.add_argument("--rerun-rho", action="store_true")
    p.add_argument(
        "--full-burnin",
        type=int,
        default=None,
        help="Only for legacy posterior files missing this metadata.",
    )
    p.add_argument(
        "--full-iterations",
        type=int,
        default=None,
        help="Only for legacy posterior files missing this metadata.",
    )
    p.add_argument("--grid-size", type=int, default=301)
    p.add_argument(
        "--trace-display-max-points",
        type=int,
        default=12000,
        help="Max displayed points per trace (does not thin ESS).",
    )
    p.add_argument(
        "--trace-smooth-window",
        type=int,
        default=151,
        help="Centered moving-average window for display only; 1 disables smoothing.",
    )
    a = p.parse_args()
    if (
        a.grid_size < 5
        or a.trace_display_max_points < 1
        or a.trace_smooth_window < 1
        or not 0 <= a.rho_burnin < a.rho_iterations - 3
    ):
        p.error("Invalid grid sizes or rho iteration/burn-in settings.")
    return a


def rho_statistics(alpha, beta, z):
    w = np.stack(
        [np.ones_like(alpha), np.broadcast_to(z[:, None], alpha.shape), alpha, beta], axis=-1
    )
    return np.einsum("nki,nkj->kij", w, w)


def rho_gaussian_parameters(stats, a0, A, b0, C, B, sigma, lam, taua, taud, rho, part):
    """Exact Gaussian conditionals, using fixed-coefficient sufficient statistics."""
    s = rho * np.sqrt(lam / sigma)
    v = (1 - rho * rho) * lam
    if part == "alpha":
        xx = stats[:, :2, :2]
        xa = stats[:, :2, 2]
        xb = stats[:, :2, 3]
        rb = xb - np.einsum("kij,kj->ki", xx, np.column_stack([b0, C])) - (B + s)[:, None] * xa
        precision = (1 / sigma + s * s / v)[:, None, None] * xx + np.eye(2)[None] / (taua * sigma)[
            :, None, None
        ]
        rhs = xa / sigma[:, None] - (s / v)[:, None] * rb
    else:
        xx = stats[:, :3, :3]
        residual = stats[:, :3, 2] - np.einsum(
            "kij,kj->ki", stats[:, :3, :2], np.column_stack([a0, A])
        )
        precision = xx / v[:, None, None] + np.eye(3)[None] / (taud * lam)[:, None, None]
        rhs = (stats[:, :3, 3] - s[:, None] * residual) / v[:, None]
    return precision, rhs


def normal_from_precision(precision, rhs, rng):
    L = np.linalg.cholesky(precision)
    mu = np.linalg.solve(np.swapaxes(L, -1, -2), np.linalg.solve(L, rhs[..., None]))[..., 0]
    noise = np.linalg.solve(np.swapaxes(L, -1, -2), rng.normal(size=rhs.shape)[..., None])[..., 0]
    return mu + noise


def rho_residual_moments(stats, a0, A, b0, C, B):
    ra = np.column_stack([-a0, -A, np.ones_like(A), np.zeros_like(A)])
    rb = np.column_stack([-b0, -C, -B, np.ones_like(B)])
    aa = np.einsum("ki,kij,kj->k", ra, stats, ra)
    bb = np.einsum("ki,kij,kj->k", rb, stats, rb)
    ab = np.einsum("ki,kij,kj->k", ra, stats, rb)
    if np.any(aa < -1e-7) or np.any(bb < -1e-7):
        raise ValueError("Negative residual sum of squares; check fixed coefficient scaling.")
    return np.maximum(aa, 0), np.maximum(bb, 0), ab


def rho_variance_logpost(sigma, lam, n, aa, bb, ab, a0, A, b0, C, B, taua, taud, hp, rho):
    if np.any(sigma <= 0) or np.any(lam <= 0):
        raise ValueError("Invalid variance proposal.")
    # Constants in rho are omitted, since each rho is fixed in its own chain.
    quad = (aa / sigma + bb / lam - 2 * rho * ab / np.sqrt(sigma * lam)) / (1 - rho * rho)
    return (
        -(hp["a3"] + 2 + n / 2) * np.log(sigma)
        - (hp["b3"] + (a0 * a0 + A * A) / (2 * taua)) / sigma
        - (hp["a4"] + 2.5 + n / 2) * np.log(lam)
        - (hp["b4"] + (b0 * b0 + C * C + B * B) / (2 * taud)) / lam
        - 0.5 * quad
    )


def sample_gig_exact(power, rate, chi, rng):
    psi = 2 * rate
    if chi <= 0 or not math.isfinite(chi):
        raise ValueError("Invalid GIG conditional parameter.")
    return float(
        geninvgauss.rvs(power, np.sqrt(psi * chi), scale=np.sqrt(chi / psi), random_state=rng)
    )


def conditional_rho_fit(full, meta, z, hp, rho, index, args, info, out):
    """Same conditional target as run_rho_fixed_sensitivity_K5, now for K=7.

    Exact population Gaussian/GIG conditionals; 2D log-variance MH with the
    Jacobian. All random draws share a seeded Generator. Sufficient statistics
    remove repeated donor summation without changing the posterior target.
    """
    alpha = np.asarray(meta["alpha_pred"], dtype=float).reshape(len(z), 7)
    beta = np.asarray(meta["beta_pred"], dtype=float).reshape(len(z), 7)
    if not np.isfinite(alpha).all() or not np.isfinite(beta).all():
        raise ValueError("Non-finite alpha_pred/beta_pred.")
    required = ("a3", "b3", "a4", "b4", "a_a", "b_a", "a_b", "b_b")
    for key in required:
        if key not in hp or not math.isfinite(float(hp[key])) or hp[key] <= 0:
            raise ValueError("Missing/invalid explicit rho prior " + key)
    seed = args.rho_seed + 1009 * index
    spec = dict(
        version=VERSION,
        K=7,
        source_sha256=info["sha256"],
        fixed_coefficients_sha256=vector_hash([alpha.ravel(), beta.ravel(), z]),
        rho=rho,
        seed=seed,
        iterations=args.rho_iterations,
        burnin=args.rho_burnin,
        priors={k: hp[k] for k in required},
        proposal_sd_sigma=0.12,
        proposal_sd_lambda=0.12,
        proposal_corr=0.0,
        target="conditional_fixed_K7_alpha_pred_beta_pred",
    )
    tag = ("%+.1f" % rho).replace("+", "p").replace("-", "m").replace(".", "p")
    path = out / "rho_fits" / ("rho_" + tag + ".npz")
    cfgpath = path.with_suffix(".json")
    if path.exists() and cfgpath.exists() and not args.rerun_rho:
        saved = read_json(cfgpath)
        if saved.get("spec") != spec or saved.get("complete") is not True:
            raise ValueError("Rho cache settings changed or cache incomplete; use --rerun-rho.")
        if file_hash(path) != saved.get("npz_sha256"):
            raise ValueError("Rho cache hash mismatch; use --rerun-rho.")
        print("Reusing completed conditional rho=%+.1f" % rho, flush=True)
        with np.load(path, allow_pickle=False) as cached:
            chains = {k: cached[k].copy() for k in ("alpha0", "A", "beta0", "B", "C")}
        return chains, saved
    needed = ("Sigma", "Lambda", "tau_a", "tau_d")
    if any(k not in full for k in needed):
        raise ValueError("Rho refits require verified post-burn-in Sigma/Lambda/tau_a/tau_d.")
    rng = np.random.default_rng(seed)
    a0, A, b0, B, C = [full[k].mean(axis=0).copy() for k in ("alpha0", "A", "beta0", "B", "C")]
    sigma = full["Sigma"].mean(axis=0)
    lam = full["Lambda"].mean(axis=0)
    taua = float(full["tau_a"].mean())
    taud = float(full["tau_d"].mean())
    nit, burn = args.rho_iterations, args.rho_burnin
    chains = {k: np.empty((nit - burn, 7)) for k in ("alpha0", "A", "beta0", "B", "C")}
    aux = {k: [] for k in ("Sigma", "Lambda", "tau_a", "tau_d", "aux_saved_itr")}
    stats = rho_statistics(alpha, beta, z)
    accepted = np.zeros(7, dtype=int)
    accepted_post = np.zeros(7, dtype=int)
    begin = time.monotonic()
    for itr in tqdm(range(nit), desc="K7 conditional rho=%+.1f" % rho):
        P, h = rho_gaussian_parameters(stats, a0, A, b0, C, B, sigma, lam, taua, taud, rho, "alpha")
        draw = normal_from_precision(P, h, rng)
        a0, A = draw[:, 0], draw[:, 1]
        P, h = rho_gaussian_parameters(stats, a0, A, b0, C, B, sigma, lam, taua, taud, rho, "beta")
        draw = normal_from_precision(P, h, rng)
        b0, C, B = draw[:, 0], draw[:, 1], draw[:, 2]
        aa, bb, ab = rho_residual_moments(stats, a0, A, b0, C, B)
        logs = np.column_stack([np.log(sigma), np.log(lam)])
        proposed = logs + 0.12 * rng.normal(size=(7, 2))
        ps, pl = np.exp(proposed[:, 0]), np.exp(proposed[:, 1])
        lp = rho_variance_logpost(
            sigma, lam, len(z), aa, bb, ab, a0, A, b0, C, B, taua, taud, hp, rho
        )
        lpp = rho_variance_logpost(ps, pl, len(z), aa, bb, ab, a0, A, b0, C, B, taua, taud, hp, rho)
        accept = np.log(np.maximum(rng.random(7), np.finfo(float).tiny)) < lpp - lp + proposed.sum(
            axis=1
        ) - logs.sum(axis=1)
        sigma = np.where(accept, ps, sigma)
        lam = np.where(accept, pl, lam)
        accepted += accept
        taua = sample_gig_exact(
            hp["a_a"] - 7, hp["b_a"], float(((a0 * a0 + A * A) / sigma).sum()), rng
        )
        taud = sample_gig_exact(
            hp["a_b"] - 10.5, hp["b_b"], float(((b0 * b0 + C * C + B * B) / lam).sum()), rng
        )
        if itr >= burn:
            accepted_post += accept
            for key, val in [("alpha0", a0), ("A", A), ("beta0", b0), ("B", B), ("C", C)]:
                chains[key][itr - burn] = val
            if (itr - burn) % 20 == 0:
                for key, val in [
                    ("Sigma", sigma.copy()),
                    ("Lambda", lam.copy()),
                    ("tau_a", taua),
                    ("tau_d", taud),
                    ("aux_saved_itr", itr + 1),
                ]:
                    aux[key].append(val)
    if not all(np.isfinite(a).all() for a in chains.values()):
        raise ValueError("Non-finite conditional sensitivity chain.")
    tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **chains, **{k: np.asarray(v) for k, v in aux.items()})
    tmp.replace(path)
    saved = dict(
        spec=spec,
        complete=True,
        npz_sha256=file_hash(path),
        runtime_seconds=time.monotonic() - begin,
        block_acceptance_all=(accepted / nit).tolist(),
        block_acceptance_postburn=(accepted_post / (nit - burn)).tolist(),
    )
    write_json(cfgpath, saved)
    return chains, saved


def rho_outputs(full, meta, z, cfg, args, info, out, primary_integrated):
    """Same conditional rho refits/numerics, AoAS-styled figures only."""
    frames, pointframes, diags, statuses = [], [], [], []
    t = np.linspace(0, 1, args.grid_size)
    phi = basis(t, 7)
    figtrace, axes = plt.subplots(3, 7, figsize=(15.2, 6.6), sharex=True)
    for idx, rho in enumerate(RHOS):
        draws, status = conditional_rho_fit(
            full, meta, z, cfg["hyperparameters"], rho, idx, args, info, out
        )
        statuses.append(status)
        df, ic = integrated(draws, 7)
        df.insert(0, "rho_s", rho)
        frames.append(df)
        pt = {"rho_s": rho, "pseudotime": t}
        for name, c in effect_coefficients(draws).items():
            for stat, val in curve_summary(c, phi).items():
                pt[name + "_" + stat] = val
        pointframes.append(pd.DataFrame(pt))
        for row, (effect, chain) in enumerate(ic.items()):
            d = ess_diagnostic(effect, chain)
            d["rho_s"] = rho
            diags.append(d)
            ax = axes[row, idx]
            plot_trace_line(ax, chain, args.trace_display_max_points // 4, args.trace_smooth_window)
            if idx == 0:
                ax.set_ylabel(effect)
                panel_label(ax, "(%s)" % chr(ord("a") + row), x=-0.47, y=1.02)
            if row == 0:
                ax.set_title(r"$\rho_s=%+.1f$" % rho)
            if row == 2:
                ax.set_xlabel("Draw index")
            apply_trace_axis_style(ax)
        del draws

    supp = out / "supplement"
    figtrace.subplots_adjust(left=0.07, right=0.99, top=0.91, bottom=0.13, wspace=0.42, hspace=0.26)
    save_figure(
        figtrace, supp / "rho_conditional_traces_K7.pdf", supp / "rho_conditional_traces_K7.png"
    )
    rdf = pd.concat(frames, ignore_index=True)
    pdf = pd.concat(pointframes, ignore_index=True)
    csv(rdf, supp / "rho_sensitivity_integrated_K7.csv")
    csv(pdf, supp / "rho_sensitivity_pointwise_K7.csv")
    csv(pd.DataFrame(diags), supp / "rho_sensitivity_diagnostics_K7.csv")
    ref = rdf[rdf.rho_s == 0].merge(
        primary_integrated, on="effect", suffixes=("_conditional_rho0", "_full_K7")
    )
    ref["conditional_minus_full"] = ref.posterior_mean_conditional_rho0 - ref.posterior_mean_full_K7
    csv(ref, supp / "rho0_vs_full_primary_K7.csv")

    # Mean effect curves: line type carries the rho group (no color reliance).
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7), sharex=True)
    for j, (ax, effect, title) in enumerate(
        zip(axes, ("ME", "DE", "TE"), ("Mediated effect", "Direct effect", "Total effect"))
    ):
        for rho in [r for r in RHOS if r != 0] + [0.0]:
            q = pdf[pdf.rho_s == rho]
            ax.plot(
                q.pseudotime,
                q[effect + "_mean"],
                label=r"$\rho_s=%+.1f$" % rho,
                zorder=6 if rho == 0 else 2,
                **RHO_STYLES[rho],
            )
        ax.axhline(0, color="0.76", linestyle="--", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Pseudotime")
        ax.set_ylabel(r"$%s(t)$" % effect)
        apply_curve_axis_style(ax)
        panel_label(ax, "(%s)" % chr(ord("a") + j))
    handles, labels = axes[0].get_legend_handles_labels()
    lookup = dict(zip(labels, handles))
    fig.legend(
        [lookup[r"$\rho_s=%+.1f$" % rho] for rho in RHOS],
        [r"$\rho_s=%+.1f$" % rho for rho in RHOS],
        loc="lower center",
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        handlelength=2.6,
        columnspacing=0.95,
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.27, wspace=0.34)
    save_figure(
        fig,
        supp / "rho_sensitivity_effect_curves_K7.pdf",
        supp / "rho_sensitivity_effect_curves_K7.png",
    )

    # Credible intervals as vertical lines, not asymmetrical error bars:
    # the posterior mean can lie outside an equal-tailed CrI when skewed.
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), sharex=True)
    for j, (ax, effect) in enumerate(zip(axes, ("IME", "IDE", "ITE"))):
        q = rdf[rdf.effect == effect].sort_values("rho_s")
        x = q.rho_s.to_numpy()
        y = q.posterior_mean.to_numpy()
        lo = q["cri_2.5"].to_numpy()
        hi = q["cri_97.5"].to_numpy()
        ax.vlines(x, lo, hi, color="0.36", linewidth=1.05, zorder=2)
        ax.hlines(lo, x - 0.009, x + 0.009, color="0.36", linewidth=1.05)
        ax.hlines(hi, x - 0.009, x + 0.009, color="0.36", linewidth=1.05)
        ax.plot(
            x,
            y,
            color="0.08",
            lw=1.05,
            marker="o",
            ms=4,
            markerfacecolor="white",
            markeredgecolor="0.08",
            markeredgewidth=0.9,
            zorder=3,
        )
        ax.axhline(0, color="0.76", ls="--", lw=0.8)
        ax.set_xlabel(r"$\rho_s$")
        ax.set_ylabel(effect)
        ax.set_title(effect)
        ax.tick_params(length=3.5, width=0.8)
        panel_label(ax, "(%s)" % chr(ord("a") + j))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.19, wspace=0.34)
    save_figure(
        fig,
        supp / "rho_sensitivity_integrated_effects_K7.pdf",
        supp / "rho_sensitivity_integrated_effects_K7.png",
    )
    return rdf, ref, statuses


def main():
    args = parse_args()
    root, cfg, run, sel, table, target, paths, data_path, cv_paths = load_selection(args)
    z, counts, state, sample, bench = load_data(data_path, cfg)
    if not (target.n_donors == len(z)).all() or not (target.n_cells == int(counts.sum())).all():
        raise ValueError("CV summary and raw-data sample sizes disagree.")
    full, meta, info = read_posterior(paths[7], 7, args, primary=True)
    info["time_alignment"] = verify_times(meta, 7, counts, state, run)
    for key in ("alpha_pred", "beta_pred"):
        a = np.asarray(meta.get(key, []), dtype=float)
        if a.shape not in ((len(z), 7), (len(z), 1, 7)) or not np.isfinite(a).all():
            raise ValueError("Missing/malformed K7 " + key)
    if args.rho_mode == "refit":
        for k in ("Sigma", "Lambda", "tau_a", "tau_d"):
            if k not in full:
                raise ValueError(
                    "Missing verified auxiliary posterior " + k + " for conditional rho refits."
                )
        if not cfg.get("hyperparameters"):
            raise ValueError(
                "Conditional rho refits require explicit hyperparameters in the original configuration."
            )
    if args.check_only:
        for k, path in sorted(paths.items()):
            if k != 7:
                other, othermeta, _ = read_posterior(path, k, args)
                verify_times(othermeta, k, counts, state, run)
                del other, othermeta
                gc.collect()
        print(
            "\nCHECK PASSED. Selected K=7; data, T_pred, dimensions and retained-draw indexing verified."
        )
        print(sample.to_string(index=False))
        print("No output tables or MCMC fits were created.")
        return
    requested = Path(args.output)
    requested = requested if requested.is_absolute() else root / requested
    out = prepare_output(
        requested,
        args.overwrite,
        [data_path, *paths.values(), *cv_paths.values(), Path(__file__).resolve()],
    )
    manifest = dict(
        generator="summarize_application",
        version=VERSION,
        complete=False,
        selection=sel,
        primary_posterior=info,
        basis_posteriors={"7": info},
        data_path=str(data_path),
        data_sha256=run["data_sha256"],
        script_sha256=file_hash(__file__),
        pointwise_grid_size=args.grid_size,
        basis_sensitivity=not args.skip_basis,
        rho_mode=args.rho_mode,
        generated_files=[],
        intervals="equal-tailed pointwise; conditional on selected K",
    )
    write_json(out / "regeneration_manifest.json", manifest)
    for name, path in cv_paths.items():
        shutil.copy2(path, out / "provenance" / name)
    write_json(
        out / "provenance" / "resolved_analysis_config.json", dict(config=cfg, arguments=vars(args))
    )
    csv(sample, out / "main" / "application_sample_summary.csv")
    csv(bench, out / "main" / "donor_average_benchmark.csv")
    print("Generating primary K=7 curves, credible intervals and diagnostics...", flush=True)
    edf, gint, shape, diag = primary_outputs(full, info, args, out)
    selection_outputs(table, target, sel, out)
    if not args.skip_basis:
        basis_outputs(full, paths, args, counts, state, run, out, manifest)
    rdf = ref = None
    if args.rho_mode == "refit":
        rdf, ref, statuses = rho_outputs(full, meta, z, cfg, args, info, out, edf)
        manifest["rho_fits"] = statuses
    manifest["complete"] = True
    manifest["generated_files"] = [
        str(p.relative_to(out))
        for p in sorted(out.rglob("*"))
        if p.is_file() and "rho_fits" not in p.parts and p.name != "regeneration_manifest.json"
    ]
    write_json(out / "regeneration_manifest.json", manifest)
    print("\nK=7 INTEGRATED EFFECTS\n" + edf.to_string(index=False))
    print("\nDone:", out)
    if rdf is None:
        print("Conditional rho sensitivity was skipped explicitly.")


# AoAS figure helpers, shared by all figure-producing branches.
def apply_trace_axis_style(ax):
    ax.tick_params(length=3.2, width=0.8)


def rolling_mean(x, window):
    """Centered moving average for display only (never for ESS/MCSE)."""
    x = np.asarray(x, dtype=float)
    if not len(x):
        return x.copy()
    w = max(1, int(window))
    if w <= 1:
        return x.copy()
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w, dtype=float)
    return np.convolve(x, kernel, mode="same") / np.convolve(np.ones(len(x)), kernel, mode="same")


def plot_trace_line(ax, chain, display_max_points=12000, smooth_window=151):
    """Original post-burn-in draws in gray, optional display-only black trend."""
    chain = np.asarray(chain, dtype=float).reshape(-1)
    step = max(1, math.ceil(len(chain) / max(1, display_max_points)))
    x = np.arange(len(chain))[::step]
    y = chain[::step]
    ax.plot(x, y, color="0.72", linewidth=0.34, rasterized=True)
    if smooth_window > 1:
        ax.plot(x, rolling_mean(y, smooth_window), color="0.08", linewidth=0.95, zorder=4)


def plot_trace_family(
    values,
    family_name,
    component_labels,
    out_pdf,
    out_png,
    display_max_points=12000,
    smooth_window=151,
):
    """One trace per basis component."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(component_labels) != values.shape[1]:
        raise ValueError("Trace family requires draw x K with K component labels.")
    K = values.shape[1]
    fig, axes = plt.subplots(1, K, figsize=(2.25 * K + 0.6, 2.65), sharex=True)
    axes = np.atleast_1d(axes)
    for k, ax in enumerate(axes):
        plot_trace_line(ax, values[:, k], display_max_points, smooth_window)
        ax.set_title(component_labels[k])
        ax.set_xlabel("Draw index")
        if k == 0:
            ax.set_ylabel(family_name)
        apply_trace_axis_style(ax)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.86, bottom=0.19, wspace=0.32)
    save_figure(fig, out_pdf, out_png)


def plot_combined_abc_traces(full, out_pdf, out_png, display_max_points=12000, smooth_window=151):
    """Optional AoAS-style combined 3 x K trace plot."""
    families = [
        (full["A"], r"$A_k$", "(a)"),
        (full["B"], r"$B_{kk}$", "(b)"),
        (full["C"], r"$C_k$", "(c)"),
    ]
    K = full["A"].shape[1]
    fig, axes = plt.subplots(3, K, figsize=(2 * K + 0.8, 6.6), sharex=True)
    for row, (values, ylabel, label) in enumerate(families):
        for k in range(K):
            ax = axes[row, k]
            plot_trace_line(ax, values[:, k], display_max_points, smooth_window)
            if row == 0:
                ax.set_title(r"$k=%d$" % (k + 1))
            if k == 0:
                ax.set_ylabel(ylabel)
                panel_label(ax, label, x=-0.43, y=1.02)
            if row == 2:
                ax.set_xlabel("Retained draw index")
            apply_trace_axis_style(ax)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.10, wspace=0.30, hspace=0.28)
    save_figure(fig, out_pdf, out_png)


def save_figure(fig, out_pdf, out_png, dpi=600):
    """Save publication-quality vector PDF and high-resolution PNG."""
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax, label, x=-0.2, y=1.05):
    """AoAS-style panel label placed just outside upper-left corner."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )


def apply_curve_axis_style(ax):
    """Axis formatting for pseudotime-domain plots."""
    ax.set_xlim(0.0, 1.0)
    ax.tick_params(length=3.5, width=0.8)


def plot_primary_figure(
    t,
    effect_summaries,
    traj_summaries,
    out_pdf,
    out_png,
):
    """
    AoAS main-text application figure.

    Visual hierarchy intentionally follows the manuscript's existing
    basis-complexity figure:
      - grayscale;
      - line style, not color, carries most group information;
      - posterior mean is visually dominant;
      - credible bands remain secondary;
      - no top/right panel borders.
    """

    genotype_styles = {
        0: {
            "color": "0.08",
            "linestyle": "-",
            "linewidth": 1.95,
            "zorder": 4,
        },
        1: {
            "color": "0.35",
            "linestyle": (0, (7, 2)),
            "linewidth": 1.75,
            "zorder": 3,
        },
        2: {
            "color": "0.60",
            "linestyle": (0, (2, 1.2)),
            "linewidth": 1.70,
            "zorder": 2,
        },
    }

    fig = plt.figure(figsize=(11.8, 6.9))

    gs = fig.add_gridspec(
        2,
        6,
        height_ratios=[1.0, 1.0],
        hspace=0.49,
        wspace=0.72,
    )

    ax_rps26 = fig.add_subplot(gs[0, 0:3])
    ax_bax = fig.add_subplot(gs[0, 3:6])

    ax_me = fig.add_subplot(gs[1, 0:2])
    ax_de = fig.add_subplot(gs[1, 2:4])
    ax_te = fig.add_subplot(gs[1, 4:6])

    # --------------------------------------------------------------
    # Mean trajectories
    # --------------------------------------------------------------
    for ax, gene, p_label in [
        (ax_rps26, "RPS26", "(a)"),
        (ax_bax, "BAX", "(b)"),
    ]:
        legend_handles = []
        legend_labels = []

        for z in [0, 1, 2]:
            summary = traj_summaries[gene][z]
            style = genotype_styles[z]

            (line,) = ax.plot(
                t,
                summary["mean"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                zorder=style["zorder"],
            )

            ax.fill_between(
                t,
                summary["low"],
                summary["high"],
                color=style["color"],
                alpha=0.11,
                linewidth=0.0,
                zorder=1,
            )

            legend_handles.append(line)
            legend_labels.append(r"$Z=%d$" % z)

        ax.set_title(gene, pad=32)
        ax.set_xlabel("Pseudotime")
        ax.set_ylabel("SCTransform residual")

        ax.legend(
            legend_handles,
            legend_labels,
            frameon=False,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.105),
            handlelength=3.0,
            columnspacing=1.25,
        )

        apply_curve_axis_style(ax)
        panel_label(ax, p_label)

    # --------------------------------------------------------------
    # Mediation effects
    # --------------------------------------------------------------
    for ax, name, title, p_label in [
        (ax_me, "ME", "Mediated effect", "(c)"),
        (ax_de, "DE", "Direct effect", "(d)"),
        (ax_te, "TE", "Total effect", "(e)"),
    ]:
        summary = effect_summaries[name]

        ax.fill_between(
            t,
            summary["low"],
            summary["high"],
            color="0.62",
            alpha=0.22,
            linewidth=0.0,
            zorder=1,
        )

        ax.plot(
            t,
            summary["mean"],
            color="0.08",
            linestyle="-",
            linewidth=1.95,
            zorder=3,
        )

        ax.axhline(
            0.0,
            color="0.75",
            linestyle="--",
            linewidth=0.8,
            zorder=0,
        )

        ax.set_title(title)
        ax.set_xlabel("Pseudotime")
        ax.set_ylabel(r"$%s(t)$" % name)

        apply_curve_axis_style(ax)
        panel_label(ax, p_label)

    fig.subplots_adjust(
        left=0.07,
        right=0.985,
        top=0.90,
        bottom=0.10,
    )

    save_figure(
        fig,
        out_pdf,
        out_png,
        dpi=600,
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError, FileExistsError) as exc:
        print("ERROR:", exc, file=sys.stderr)
        sys.exit(1)
