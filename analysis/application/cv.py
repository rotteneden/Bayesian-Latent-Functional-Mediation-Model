"""Conditional cross-validation for the OneK1K analysis using donor and cell identities."""

from __future__ import annotations
from blafm.mcmc import (
    compute_bspline_basis_ragged,
    update_Lambda,
    update_Sigma,
    update_alpha0_A,
    update_beta0_BC_diag,
    update_tau_a,
    update_tau_d,
)
from analysis.conditional_cv import (
    build_fixed_test_basis,
    file_hash,
    integer,
    jsonable,
    object_hash,
    read_json,
    save_csv,
    save_json,
    split_one_fold,
    summarize_fold_scores,
    update_conditional_log_accumulators,
    vector_hash,
    write_trace,
)
from analysis import conditional_cv
from blafm import mcmc

import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

for _thread_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_name, "1")
import numpy as np
import scipy
from scipy.stats import invgamma

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(values, **kwargs):
        print(kwargs.get("desc", "MCMC"), flush=True)
        return values


VERSION = "1.0.1"
METHOD = "application_scheme2_fixed_own_K_T_pred"
SCORE_TARGET = "conditional_fixed_own_K_T_pred"
LOG_2PI = math.log(2.0 * math.pi)
HP_KEYS = ("a1", "b1", "a2", "b2", "a3", "b3", "a4", "b4", "a_a", "b_a", "a_b", "b_b")
COMPONENTS = ("lpd_x", "lpd_y_given_x", "lpd_xy")
# Upper-level Gibbs and spline routines below are reused from the supplied code.


def ragged(raw, label):
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{label}: expected a nonempty list of donor rows.")
    rows = [np.asarray(row, dtype=float) for row in raw]
    if any(r.ndim != 1 or r.size == 0 or not np.isfinite(r).all() for r in rows):
        raise ValueError(f"{label}: donor rows must be nonempty finite vectors.")
    return rows


def normalize_ids(donor_ids, cell_ids, counts, label):
    if donor_ids is None and cell_ids is None:
        return None
    if not isinstance(donor_ids, list) or not isinstance(cell_ids, list):
        raise ValueError(f"{label}: provide both donor_ids and donor-major cell_ids.")
    if len(donor_ids) != len(counts) or len(cell_ids) != len(counts):
        raise ValueError(f"{label}: ID donor count mismatch.")
    if any(v is None or isinstance(v, (list, dict)) for v in donor_ids):
        raise ValueError(f"{label}: invalid donor ID.")
    d = [str(v) for v in donor_ids]
    if len(set(d)) != len(d) or any(not v for v in d):
        raise ValueError(f"{label}: duplicate or empty donor ID.")
    c = []
    for row, count in zip(cell_ids, counts):
        if not isinstance(row, list) or len(row) != count:
            raise ValueError(f"{label}: cell ID row does not match actual cell count.")
        if any(v is None or isinstance(v, (list, dict)) for v in row):
            raise ValueError(f"{label}: invalid cell ID.")
        s = [str(v) for v in row]
        if len(set(s)) != len(s) or any(not v for v in s):
            raise ValueError(f"{label}: duplicate/empty cell ID within donor.")
        c.append(s)
    return d, c


def load_application_data(config):
    path = config["data_json"]
    doc = read_json(path)
    keys = {
        "x": "x",
        "y": "y",
        "z": "z",
        "cell_type": "cell_type",
        "donor_ids": "donor_ids",
        "cell_ids": "cell_ids",
        **config.get("data_keys", {}),
    }
    X = ragged(doc[keys["x"]], "x")
    Y = ragged(doc[keys["y"]], "y")
    state = ragged(doc[keys["cell_type"]], "cell_type")
    counts = np.asarray([len(x) for x in X], dtype=int)
    if any([len(r) for r in a] != counts.tolist() for a in (Y, state)):
        raise ValueError("x/y/cell_type donor-cell shapes differ.")
    if np.any(counts < 2):
        raise ValueError(
            "Cell CV requires at least two cells per donor; no cells are silently dropped."
        )
    if any(not np.isin(r, [0, 1]).all() for r in state):
        raise ValueError("cell_type must use the same 0=early,1=late encoding as the full model.")
    z = np.asarray(doc[keys["z"]], dtype=float)
    if z.ndim not in (1, 2) or z.size != len(counts):
        raise ValueError("z must contain one scalar genotype per donor.")
    z = z.reshape(-1) + float(config["z_offset"])
    if not np.isfinite(z).all():
        raise ValueError("Nonfinite genotype.")
    ids = normalize_ids(doc.get(keys["donor_ids"]), doc.get(keys["cell_ids"]), counts, "data")
    donor_ids, cell_ids = (
        ids
        if ids
        else (
            [f"donor_index:{i}" for i in range(len(counts))],
            [[f"cell_index:{c}" for c in range(n)] for n in counts],
        )
    )
    return SimpleNamespace(
        X=X,
        Y=Y,
        Z=z,
        state=[r.astype(np.int8) for r in state],
        counts=counts,
        folder=path.parent,
        path=path,
        ids=ids,
        donor_ids=donor_ids,
        cell_ids=cell_ids,
        data_sha256=file_hash(path),
        identity_sha256=object_hash([donor_ids, cell_ids]),
    )


def posterior_fields(path):
    """Read only small metadata and T_pred, skipping allocations for full chains."""
    wanted = {
        "init",
        "T_pred",
        "K",
        "donor_ids",
        "cell_ids",
        "data_sha256",
        "rho_fixed",
        "rho",
        "alpha_pred",
        "beta_pred",
        "A_HAT",
        "C_HAT",
        "B_HAT",
    }
    try:
        import ijson
        from ijson.common import ObjectBuilder
    except ImportError:
        if path.stat().st_size > 256 * 1024**2:
            raise ImportError("Large posterior JSON: install ijson (python -m pip install ijson).")
        doc = read_json(path)
        return {k: doc[k] for k in wanted if k in doc}
    out = {}
    builder = None
    target = None
    depth = 0
    with path.open("rb") as handle:
        for prefix, event, value in ijson.parse(handle, use_float=True):
            if builder is not None:
                builder.event(event, value)
                depth += int(event in ("start_map", "start_array"))
                depth -= int(event in ("end_map", "end_array"))
                if depth == 0:
                    out[target] = builder.value
                    builder = None
                    target = None
            elif prefix in wanted and event != "map_key":
                if event in ("start_map", "start_array"):
                    target = prefix
                    builder = ObjectBuilder()
                    depth = 1
                    builder.event(event, value)
                elif event not in ("end_map", "end_array"):
                    out[prefix] = value
    return out


def validate_source_K(K, meta, fitcfg, n_donors, path):
    """Validate saved K and small coefficient means; never trust a folder name."""
    init = meta.get("init", {})
    evidence = {}
    for label, obj in (("init.K", init), ("posterior.K", meta), ("fit_config.K", fitcfg)):
        if "K" in obj:
            evidence[label] = integer(obj["K"], label)
    n_metadata = len(evidence)
    for name in ("alpha_pred", "beta_pred", "A_HAT", "C_HAT", "B_HAT"):
        if name not in meta:
            continue
        try:
            a = np.asarray(meta[name], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}: {name} must be a rectangular coefficient-mean array."
            ) from exc
        if a.size == 0 or not np.isfinite(a).all():
            raise ValueError(f"{path}: {name} is empty or nonfinite.")
        if name in ("alpha_pred", "beta_pred"):
            if a.ndim != 2 or a.shape[0] != n_donors:
                raise ValueError(
                    f"{path}: {name} has shape {a.shape}; expected (n_donors={n_donors}, K)."
                )
            inferred_K = a.shape[1]
        elif name == "B_HAT":
            if a.ndim != 2 or a.shape[0] != a.shape[1]:
                raise ValueError(
                    f"{path}: B_HAT has shape {a.shape}; expected a square K-by-K matrix."
                )
            inferred_K = a.shape[0]
        else:
            if not (a.ndim == 1 or (a.ndim == 2 and 1 in a.shape)):
                raise ValueError(f"{path}: {name} has shape {a.shape}; expected a length-K vector.")
            inferred_K = a.size
        evidence[f"{name}.dimension"] = int(inferred_K)
    if not evidence:
        raise ValueError(
            f"{path}: cannot verify expected K={K}: no K metadata or recognized "
            "coefficient means (alpha_pred, beta_pred, A_HAT, C_HAT, B_HAT). "
            "T_pred and the folder name alone do not identify the basis dimension."
        )
    details = ", ".join(f"{name}={value}" for name, value in evidence.items())
    if any(value != K for value in evidence.values()):
        raise ValueError(
            f"{path}: expected K={K}, but source evidence is [{details}]. "
            "Check the actual fit; do not relabel the posterior file to bypass this mismatch."
        )
    mode = (
        "metadata_and_or_coefficient_dimensions"
        if n_metadata
        else "coefficient_dimensions_no_K_metadata"
    )
    print(f"  Verified K={K}: {details}", flush=True)
    return {"mode": mode, "evidence": evidence}


def load_source(K, path, dat, config):
    print(f"Reading K={K}: {path}", flush=True)
    meta = posterior_fields(path)
    if "T_pred" not in meta:
        raise KeyError(f"{path}: missing exact key T_pred.")
    init = meta.get("init", {})
    if not isinstance(init, dict):
        raise ValueError(f"{path}: init must be a mapping.")
    fitcfg_path = path.parent / "fit_config.json"
    fitcfg = read_json(fitcfg_path) if fitcfg_path.exists() else {}
    K_validation = validate_source_K(K, meta, fitcfg, len(dat.counts), path)
    if "fold" in fitcfg:
        raise ValueError(f"{path}: fold fit detected; provide the FULL-data fit.")
    for obj in (meta, init, fitcfg):
        for name in ("rho", "rho_fixed"):
            if name in obj and float(obj[name]) != 0.0:
                raise ValueError("This script implements the supplied rho=0 model only.")
        if "degree" in obj and integer(obj["degree"], "degree") != 3:
            raise ValueError("Full fit must use cubic (degree=3) B-splines.")
        if "z_offset" in obj and float(obj["z_offset"]) != float(config["z_offset"]):
            raise ValueError("Source z_offset disagrees with configuration.")
    declared = config.get("hyperparameters")
    hp_sources = [init, fitcfg.get("hyperparameters", {})]
    if declared is not None:
        if not isinstance(declared, dict) or set(declared) != set(HP_KEYS):
            raise ValueError(f"hyperparameters must supply exactly {HP_KEYS}.")
        hp_sources.append(declared)
    hp = {}
    for name in HP_KEYS:
        vals = [float(obj[name]) for obj in hp_sources if name in obj]
        if not vals or any(not math.isfinite(v) or v <= 0 for v in vals):
            raise ValueError(
                f"{path}: missing/invalid {name}; specify hyperparameters explicitly if needed."
            )
        if any(v != vals[0] for v in vals):
            raise ValueError(f"{path}: conflicting prior hyperparameter {name}.")
        hp[name] = vals[0]
    raw = ragged(meta["T_pred"], f"{path}:T_pred")
    if len(raw) != len(dat.counts):
        raise ValueError("T_pred donor count differs from application data.")
    sidepath = config.get("source_index_files", {}).get(str(K))
    side = read_json(sidepath) if sidepath else {}
    for obj in (meta, fitcfg, side):
        h = obj.get("data_sha256")
        if isinstance(h, str) and h != dat.data_sha256:
            raise ValueError(f"{path}: recorded data SHA256 differs from current data.")
    src_d = side.get("donor_ids", meta.get("donor_ids"))
    src_c = side.get("cell_ids", meta.get("cell_ids"))
    if src_d is not None or src_c is not None:
        if not isinstance(src_c, list) or any(not isinstance(r, list) for r in src_c):
            raise ValueError("Source cell_ids must be ragged donor-major lists of actual cells.")
        source_ids = normalize_ids(src_d, src_c, [len(r) for r in src_c], "source")
        if len(src_c) != len(raw):
            raise ValueError("Source IDs/T_pred donor count mismatch.")
    else:
        source_ids = None
    if dat.ids and source_ids:
        src_d, src_c = source_ids
        if set(src_d) != set(dat.donor_ids):
            raise ValueError("Source and data donor ID sets differ.")
        lookup = {d: i for i, d in enumerate(src_d)}
        aligned = []
        for i, d in enumerate(dat.donor_ids):
            j = lookup[d]
            nc = len(src_c[j])
            if len(raw[j]) not in (nc, int(dat.counts.max())):
                raise ValueError("Source T_pred padding/count mismatch.")
            if set(src_c[j]) != set(dat.cell_ids[i]):
                raise ValueError(f"Donor {d}: source/data cell ID sets differ.")
            position = {c: p for p, c in enumerate(src_c[j])}
            aligned.append(raw[j][[position[c] for c in dat.cell_ids[i]]].copy())
        alignment = "matched_donor_and_cell_ids"
    else:
        if not config["assume_same_cell_order"]:
            raise ValueError(
                "Comparable donor/cell IDs are unavailable. After checking that all "
                "fits used EXACTLY this data file and row order, set assume_same_cell_order=true "
                "in the configuration; otherwise provide matching IDs/source_index_files."
            )
        widths = np.asarray([len(r) for r in raw])
        if not (np.array_equal(widths, dat.counts) or np.all(widths == dat.counts.max())):
            raise ValueError("T_pred must be ragged by cell counts or padded to their maximum.")
        aligned = [r[:n].copy() for r, n in zip(raw, dat.counts)]
        alignment = "same_order_explicitly_declared_NOT_verified_by_ids"
    for i, (r, s) in enumerate(zip(aligned, dat.state)):
        if np.any(r < 0.5 * s) or np.any(r > 0.5 * s + 0.5):
            raise ValueError(f"K={K}, donor index {i}: T_pred is outside its cell_type interval.")
        r.flags.writeable = False
    source = {
        "path": str(path),
        "key": "T_pred",
        "source_K": K,
        "trimmed_times_sha256": vector_hash(aligned),
        "file_size": path.stat().st_size,
        "metadata_sha256": object_hash({k: v for k, v in meta.items() if k != "T_pred"}),
        "fit_config_sha256": file_hash(fitcfg_path) if fitcfg_path.exists() else None,
        "source_index_sha256": file_hash(sidepath) if sidepath else None,
        "alignment": alignment,
        "K_validation": K_validation,
        "hyperparameters": hp,
        "reference_uses_heldout_expression": True,
    }
    return aligned, source


def make_folds(dat, config):
    rng = np.random.default_rng(config["fold_seed"])
    F = config["n_folds"]
    assignments = []
    for n, state in zip(dat.counts, dat.state):
        labels = rng.permutation(F)
        f = np.full(n, -1, dtype=np.int16)
        start = 0
        groups = (
            [np.flatnonzero(state == s) for s in (0, 1)]
            if config["fold_stratification"] == "donor_cell_type"
            else [np.arange(n)]
        )
        for ind in groups:
            ind = rng.permutation(ind)
            f[ind] = labels[(np.arange(len(ind)) + start) % F]
            start += len(ind)
        if np.any(f < 0) or any(np.all(f == j) for j in range(F)):
            raise ValueError("A donor has no training cells in a fold.")
        assignments.append(f)
    if set(np.concatenate(assignments).tolist()) != set(range(F)):
        raise ValueError("At least one CV fold is globally empty.")
    return assignments


def sufficient_statistics(X, Y, T, K):
    phi = compute_bspline_basis_ragged(T, np.linspace(0, 1, K - 2), 3)
    return {
        "gram": np.asarray([p.T @ p for p in phi]),
        "xp": np.asarray([np.asarray(x) @ p for x, p in zip(X, phi)]),
        "yp": np.asarray([np.asarray(y) @ p for y, p in zip(Y, phi)]),
        "xx": np.asarray([np.dot(x, x) for x in X]),
        "yy": np.asarray([np.dot(y, y) for y in Y]),
        "counts": np.asarray([len(t) for t in T]),
    }


def update_coefficients_cached(alpha0, A, B, beta0, C, Z, stats, Sigma, Lambda, sigma_i, tau_i):
    """Same block normal full conditional as update_alpha_beta_block, batched."""
    n, K = stats["xp"].shape
    Si = np.linalg.inv(Sigma)
    Li = np.linalg.inv(Lambda)
    Q = np.empty((n, 2 * K, 2 * K))
    Q[:, :K, :K] = Si + stats["gram"] / sigma_i[:, None, None] + B.T @ Li @ B
    Q[:, K:, K:] = Li + stats["gram"] / tau_i[:, None, None]
    Q[:, :K, K:] = -B.T @ Li
    Q[:, K:, :K] = -Li @ B
    ma = alpha0 + A * Z[:, None]
    mb = beta0 + C * Z[:, None]
    qa = stats["xp"] / sigma_i[:, None] + ma @ Si.T - mb @ (B.T @ Li).T
    qb = stats["yp"] / tau_i[:, None] + mb @ Li.T
    q = np.concatenate([qa, qb], axis=1)
    L = np.linalg.cholesky(Q)
    Lt = np.swapaxes(L, -1, -2)
    mu = np.linalg.solve(Lt, np.linalg.solve(L, q[..., None]))[..., 0]
    eps = np.linalg.solve(Lt, np.random.normal(size=(n, 2 * K, 1)))[..., 0]
    draw = mu + eps
    return draw[:, :K], draw[:, K:]


def update_variance_cached(coef, stats, which, shape, scale):
    xp = stats["xp" if which == "x" else "yp"]
    xx = stats["xx" if which == "x" else "yy"]
    cross = 2 * np.sum(coef * xp, axis=1)
    quad = np.einsum("ni,nij,nj->n", coef, stats["gram"], coef)
    sse = xx - cross + quad
    tolerance = 1e-10 * np.maximum(1.0, np.abs(xx) + np.abs(cross) + np.abs(quad))
    if np.any(sse < -tolerance):
        raise FloatingPointError("Negative residual sum of squares beyond rounding tolerance.")
    return np.atleast_1d(
        invgamma.rvs(a=shape + stats["counts"] / 2, scale=scale + np.maximum(sse, 0) / 2)
    )


CELL_FIELDS = [
    "K",
    "fold",
    "donor_index",
    "donor_id",
    "cell_index",
    "cell_id",
    "cell_type",
    "X",
    "Y",
    *COMPONENTS,
    "n_score_draws",
    "t_hat",
    "time_source_K",
]


def write_cells(path, heldout, dat, K, fold, n_score, lpd_x, lpd_xy):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CELL_FIELDS)
        writer.writeheader()
        for j, (i, c) in enumerate(zip(heldout["donor_index"], heldout["cell_index"])):
            writer.writerow(
                {
                    "K": K,
                    "fold": fold,
                    "donor_index": int(i),
                    "donor_id": dat.donor_ids[i],
                    "cell_index": int(c),
                    "cell_id": dat.cell_ids[i][c],
                    "cell_type": int(heldout["state"][j]),
                    "X": float(heldout["x"][j]),
                    "Y": float(heldout["y"][j]),
                    "lpd_x": float(lpd_x[j]),
                    "lpd_y_given_x": float(lpd_xy[j] - lpd_x[j]),
                    "lpd_xy": float(lpd_xy[j]),
                    "n_score_draws": n_score,
                    "t_hat": float(heldout["t_hat"][j]),
                    "time_source_K": K,
                }
            )


def fit_one(dat, times, assignments, job, folder):
    K, fold = job["K"], job["fold"]
    s = SimpleNamespace(**job["settings"])
    hp = s.hyperparameters
    if (
        job["fixed_time_source"]["source_K"] != K
        or vector_hash(times) != job["fixed_time_source"]["trimmed_times_sha256"]
    ):
        raise ValueError("Candidate-specific times do not match the validated source.")
    X, Y, states, heldout = split_one_fold(dat.X, dat.Y, dat.state, assignments, fold - 1)
    T = [t[f != fold - 1].copy() for t, f in zip(times, assignments)]
    for t in T:
        t.flags.writeable = False
    before = vector_hash(T)
    stats = sufficient_statistics(X, Y, T, K)
    for value in stats.values():
        value.flags.writeable = False
    test_basis = build_fixed_test_basis(heldout, times, np.linspace(0, 1, K - 2), 3)
    np.random.seed(s.seed)
    rng = np.random.default_rng(s.seed)
    n = len(T)
    alpha0 = np.zeros((1, K))
    A = np.full((1, K), 0.1)
    beta0 = np.zeros((1, K))
    C = np.full((1, K), 0.1)
    B = np.eye(K) * 0.1
    Sigma = np.eye(K)
    Lambda = np.eye(K)
    sigma_i = np.full(n, 0.1)
    tau_i = np.full(n, 0.1)
    tau_a = 10.0
    tau_d = 10.0
    acc_x = np.full(len(heldout["x"]), -np.inf)
    acc_xy = acc_x.copy()
    sums = {}
    trace = []
    n_retained = 0
    n_score = 0
    expected_S = (s.iterations - s.burnin + s.score_thin - 1) // s.score_thin
    draws = None
    if job["save_scoring_draws"]:
        draws = {name: np.empty((expected_S, n, K)) for name in ("alpha", "beta")}
        draws.update({name: np.empty((expected_S, n)) for name in ("sigma_i", "tau_i")})
        draws["iteration"] = np.empty(expected_S, dtype=np.int64)
    started = time.monotonic()
    for itr in tqdm(range(s.iterations), desc=f"K={K} fold={fold} fixed T_pred"):
        alpha, beta = update_coefficients_cached(
            alpha0, A, B, beta0, C, dat.Z, stats, Sigma, Lambda, sigma_i, tau_i
        )
        sigma_i = update_variance_cached(alpha, stats, "x", hp["a1"], hp["b1"])
        tau_i = update_variance_cached(beta, stats, "y", hp["a2"], hp["b2"])
        alpha0, A = update_alpha0_A(alpha, dat.Z, Sigma, n, K, tau_a)
        theta = update_beta0_BC_diag(dat.Z, beta, alpha, Lambda, n, K, tau_d)
        beta0 = theta[:, 0].reshape(1, -1)
        C = theta[:, 1].reshape(1, -1)
        B = np.diag(theta[:, 2])
        Sigma = update_Sigma(alpha, alpha0, A, dat.Z, n, K, tau_a, hp["a3"], hp["b3"])
        Lambda = update_Lambda(alpha, beta, beta0, B, C, dat.Z, n, K, tau_d, hp["a4"], hp["b4"])
        tau_a = update_tau_a(hp["a_a"], hp["b_a"], alpha0, A, Sigma, K, rng=rng)
        tau_d = update_tau_d(hp["a_b"], hp["b_b"], beta0, C, B, Lambda, K, rng=rng)
        values = dict(
            alpha0=alpha0,
            A=A,
            beta0=beta0,
            C=C,
            B=B,
            Sigma=Sigma,
            Lambda=Lambda,
            alpha=alpha,
            beta=beta,
            sigma_i=sigma_i,
            tau_i=tau_i,
            tau_a=tau_a,
            tau_d=tau_d,
            ME_coeff=A * np.diag(B)[None, :],
            TE_coeff=C + A * np.diag(B)[None, :],
        )
        if any(not np.isfinite(value).all() for value in values.values()):
            raise FloatingPointError(
                f"K={K}, fold={fold}, iteration={itr + 1}: nonfinite MCMC state."
            )
        if itr % s.trace_thin == 0 or itr == s.iterations - 1:
            row = {
                "iteration": itr + 1,
                "post_burnin": itr >= s.burnin,
                "tau_a": float(tau_a),
                "tau_d": float(tau_d),
                "sigma_mean": float(sigma_i.mean()),
                "tau_mean": float(tau_i.mean()),
                "fixed_t00": float(T[0][0]),
            }
            for name, val in [
                ("alpha0", alpha0.ravel()),
                ("A", A.ravel()),
                ("beta0", beta0.ravel()),
                ("C", C.ravel()),
                ("B", np.diag(B)),
                ("Sigma", np.diag(Sigma)),
                ("Lambda", np.diag(Lambda)),
                ("alpha_donor0", alpha[0]),
                ("beta_donor0", beta[0]),
            ]:
                row.update({f"{name}_{k + 1}": float(v) for k, v in enumerate(val)})
            trace.append(row)
        if itr >= s.burnin:
            n_retained += 1
            for name, val in values.items():
                if name not in sums:
                    sums[name] = np.zeros_like(val)
                sums[name] += val
            if (itr - s.burnin) % s.score_thin == 0:
                update_conditional_log_accumulators(
                    heldout, test_basis, alpha, beta, sigma_i, tau_i, acc_x, acc_xy
                )
                if draws is not None:
                    for name in ("alpha", "beta", "sigma_i", "tau_i"):
                        draws[name][n_score] = values[name]
                    draws["iteration"][n_score] = itr + 1
                n_score += 1
    if vector_hash(T) != before or n_score != expected_S:
        raise RuntimeError("Fixed-time/scoring-draw invariant failed.")
    lpd_x = acc_x - math.log(n_score)
    lpd_xy = acc_xy - math.log(n_score)
    if not np.isfinite(lpd_x).all() or not np.isfinite(lpd_xy).all():
        raise FloatingPointError("Nonfinite predictive LPD.")
    summary = summarize_fold_scores(heldout, lpd_x, lpd_xy)
    summary.update(
        K=K,
        fold=fold,
        n_score_draws=n_score,
        n_retained_draws=n_retained,
        n_train_cells=int(stats["counts"].sum()),
        time_source_K=K,
        test_time_integrated=False,
        pseudotime_update_count=0,
        max_abs_training_time_change=0.0,
        fixed_training_times_sha256=before,
        elapsed_seconds=time.monotonic() - started,
    )
    write_cells(folder / "cell_lpd.csv", heldout, dat, K, fold, n_score, lpd_x, lpd_xy)
    write_trace(folder / "trace.csv", trace)
    save_json(
        folder / "posterior_summary.json",
        {name: value / n_retained for name, value in sums.items()},
    )
    save_json(folder / "fold_summary.json", summary)
    save_json(
        folder / "cv_config.json",
        {
            **job,
            "fixed_training_times_sha256": before,
            "training_times_updated": False,
            "test_time_integrated": False,
            "reference_uses_heldout_expression": True,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    files = [
        "cell_lpd.csv",
        "trace.csv",
        "posterior_summary.json",
        "fold_summary.json",
        "cv_config.json",
    ]
    if draws is not None:
        with (folder / "posterior_draws.npz").open("wb") as handle:
            np.savez_compressed(handle, **draws)
        files.append("posterior_draws.npz")
    save_json(
        folder / "_SUCCESS.json",
        {
            "method": METHOD,
            "task_fingerprint": object_hash(job),
            "output_sha256": {name: file_hash(folder / name) for name in files},
        },
    )
    print(
        f"  Finished K={K}, fold={fold}; {n_score} scoring draws; joint LPD={lpd_xy.mean():.6f}",
        flush=True,
    )


def completed_job(folder, job):
    marker = folder / "_SUCCESS.json"
    if not marker.exists():
        return False
    m = read_json(marker)
    if m.get("method") != METHOD or m.get("task_fingerprint") != object_hash(job):
        raise ValueError(f"{folder}: completed fit has incompatible inputs/settings.")
    files = [
        "cell_lpd.csv",
        "trace.csv",
        "posterior_summary.json",
        "fold_summary.json",
        "cv_config.json",
    ]
    if job["save_scoring_draws"]:
        files.append("posterior_draws.npz")
    for name in files:
        if not (folder / name).is_file() or file_hash(folder / name) != m.get(
            "output_sha256", {}
        ).get(name):
            raise ValueError(f"{folder}: completed output missing/changed: {name}.")
    return True


def validate_destination(args, folder, job):
    if not folder.exists() or not any(folder.iterdir()):
        return
    request = folder / "task_request.json"
    if not request.exists() or object_hash(read_json(request)) != object_hash(job):
        raise FileExistsError(f"{folder}: incompatible or unowned folder. Use a new output_dir.")
    if args.overwrite:
        return
    if completed_job(folder, job):
        if not args.resume:
            raise FileExistsError(f"{folder}: complete; use --resume or --overwrite.")
    elif not args.restart_incomplete:
        raise FileExistsError(
            f"{folder}: incomplete; --resume --restart-incomplete restarts this fold."
        )


def summarize(config, dat, assignments, times, jobs):
    out = config["output_dir"]
    scores = []
    donor_rows = []
    fold_rows = []
    offsets = np.r_[0, np.cumsum(dat.counts)[:-1]]
    total = int(dat.counts.sum())
    flat_donor = np.repeat(np.arange(len(dat.counts)), dat.counts)
    states = np.concatenate(dat.state)
    tmp = out / "cell_lpd_all.csv.tmp"
    complete_Ks = []
    missing = []
    with tmp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CELL_FIELDS)
        writer.writeheader()
        for K in config["candidate_Ks"]:
            seen = np.zeros(total, dtype=bool)
            lpd = np.full((total, 3), np.nan)
            finished = []
            for fold in range(1, config["n_folds"] + 1):
                job, folder = jobs[(K, fold)]
                if not completed_job(folder, job):
                    missing.append({"K": K, "fold": fold})
                    continue
                fold_rows.append(read_json(folder / "fold_summary.json"))
                finished.append(fold)
                count = 0
                with (folder / "cell_lpd.csv").open(encoding="utf-8-sig", newline="") as source:
                    for row in csv.DictReader(source):
                        i = integer(row["donor_index"], "donor")
                        c = integer(row["cell_index"], "cell")
                        if not (0 <= i < len(dat.counts) and 0 <= c < dat.counts[i]):
                            raise ValueError("Saved LPD index out of range.")
                        index = offsets[i] + c
                        if (
                            seen[index]
                            or assignments[i][c] != fold - 1
                            or int(row["K"]) != K
                            or int(row["fold"]) != fold
                            or int(row["time_source_K"]) != K
                        ):
                            raise ValueError("Saved LPD has wrong K/fold or duplicate cells.")
                        if (
                            row["donor_id"] != dat.donor_ids[i]
                            or row["cell_id"] != dat.cell_ids[i][c]
                        ):
                            raise ValueError("Saved cell IDs do not match data.")
                        if (
                            float(row["X"]) != dat.X[i][c]
                            or float(row["Y"]) != dat.Y[i][c]
                            or float(row["t_hat"]) != times[K][i][c]
                        ):
                            raise ValueError("Saved observations/time differ from current inputs.")
                        seen[index] = True
                        count += 1
                        lpd[index] = [float(row[n]) for n in COMPONENTS]
                        writer.writerow(row)
                if count != sum(int(np.sum(f == fold - 1)) for f in assignments):
                    raise ValueError("Incomplete held-out rows in a completed fold.")
            if len(finished) != config["n_folds"]:
                continue
            if not seen.all() or not np.isfinite(lpd).all():
                raise ValueError("Incomplete/nonfinite pooled LPD.")
            if not np.allclose(lpd[:, 2], lpd[:, 0] + lpd[:, 1], rtol=1e-12, atol=1e-12):
                raise ValueError("LPD component decomposition mismatch.")
            complete_Ks.append(K)
            for scope in ("all", "early", "late"):
                at = (
                    np.ones(total, dtype=bool) if scope == "all" else states == int(scope == "late")
                )
                if not at.any():
                    continue
                ids = flat_donor[at]
                ns = np.bincount(ids, minlength=len(dat.counts))
                present = ns > 0
                dmeans = np.stack(
                    [
                        np.bincount(ids, weights=lpd[at, j], minlength=len(ns))[present]
                        / ns[present]
                        for j in range(3)
                    ],
                    axis=1,
                )
                for i, dm in zip(np.flatnonzero(present), dmeans):
                    donor_rows.append(
                        dict(
                            K=K,
                            scope=scope,
                            donor_index=int(i),
                            donor_id=dat.donor_ids[i],
                            n_cells=int(ns[i]),
                            **{name: float(dm[j]) for j, name in enumerate(COMPONENTS)},
                        )
                    )
                for weighting, v in [
                    ("donor_balanced", dmeans.mean(axis=0)),
                    ("cell_weighted", lpd[at].mean(axis=0)),
                ]:
                    scores.append(
                        dict(
                            K=K,
                            scope=scope,
                            weighting=weighting,
                            n_cells=int(at.sum()),
                            n_donors=int(present.sum()),
                            n_folds=config["n_folds"],
                            **{name: float(v[j]) for j, name in enumerate(COMPONENTS)},
                        )
                    )
    tmp.replace(out / "cell_lpd_all.csv")
    save_csv(
        out / "donor_scores_by_K.csv",
        donor_rows,
        ["K", "scope", "donor_index", "donor_id", "n_cells", *COMPONENTS],
    )
    save_csv(
        out / "cv_summary_by_K.csv",
        scores,
        ["K", "scope", "weighting", "n_cells", "n_donors", "n_folds", *COMPONENTS],
    )
    save_csv(out / "fold_scores.csv", fold_rows, list(fold_rows[0]) if fold_rows else ["K", "fold"])
    selection = {
        "method": METHOD,
        "complete": not missing,
        "candidate_Ks": config["candidate_Ks"],
        "complete_Ks": complete_Ks,
        "missing_folds": missing,
        "selected_K": None,
        "scope": "all",
        "weighting": "donor_balanced",
        "component": "lpd_xy",
        "rule": "argmax at full precision; exact ties choose smallest K",
        "test_time_integrated": False,
        "reference_uses_heldout_expression": True,
        "convergence_assessed_by_script": False,
        "conditional_plugin_score": True,
    }
    primary = [r for r in scores if r["scope"] == "all" and r["weighting"] == "donor_balanced"]
    for row in primary:
        print(f"K={row['K']}: donor-balanced joint LPD={row['lpd_xy']:.9f}", flush=True)
    if not missing:
        ranked = sorted(primary, key=lambda r: (-r["lpd_xy"], r["K"]))
        best = ranked[0]
        selection.update(
            selected_K=best["K"],
            best_lpd_xy=best["lpd_xy"],
            exactly_tied_Ks=[r["K"] for r in ranked if r["lpd_xy"] == best["lpd_xy"]],
            runner_up_K=ranked[1]["K"] if len(ranked) > 1 else None,
            gap_to_runner_up=best["lpd_xy"] - ranked[1]["lpd_xy"] if len(ranked) > 1 else None,
            full_latent_T_posterior_file=str(config["posterior_files"][str(best["K"])]),
        )
        print(
            f"Selected K={best['K']} by the declared score. Check MCMC diagnostics before reporting.",
            flush=True,
        )
    else:
        print(f"{len(missing)} folds remain. No final K selection yet.", flush=True)
    save_json(out / "selection.json", selection)


def parse_args():
    p = argparse.ArgumentParser(description="Application Scheme 2: fixed own-K T_pred CV.")
    p.add_argument("--config", type=Path)
    p.add_argument(
        "--Ks", nargs="+", type=int, help="Execute only these Ks; candidate set stays frozen."
    )
    p.add_argument("--folds", nargs="+", type=int, help="Execute only these 1-based folds.")
    p.add_argument(
        "--check-only",
        "--dry-run",
        action="store_true",
        help="Validate inputs; no result writes or fits.",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--restart-incomplete", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="Rerun selected matching folds only.")
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args()
    if not args.config:
        p.error("--config is required.")
    if args.overwrite and (args.resume or args.restart_incomplete):
        p.error("--overwrite cannot be combined with --resume/--restart-incomplete.")
    if args.summarize_only and args.overwrite:
        p.error("Cannot overwrite while summarizing only.")
    return args


def read_config(path):
    c = read_json(path)
    base = path.resolve().parent
    required = {
        "data_json",
        "posterior_files",
        "output_dir",
        "candidate_Ks",
        "z_offset",
        "assume_same_cell_order",
        "n_folds",
        "fold_seed",
        "fold_stratification",
        "iterations",
        "burnin",
        "score_thin",
        "trace_thin",
        "mcmc_seed",
        "hyperparameters",
        "save_scoring_draws",
    }
    unknown = set(c) - required - {"data_keys", "source_index_files"}
    if required - set(c) or unknown:
        raise ValueError(
            f"Configuration missing keys {sorted(required - set(c))}; unknown keys {sorted(unknown)}."
        )

    def resolve(p):
        p = Path(p).expanduser()
        return (p if p.is_absolute() else base / p).resolve()

    c["data_json"] = resolve(c["data_json"])
    c["output_dir"] = resolve(c["output_dir"])
    if not isinstance(c["posterior_files"], dict):
        raise ValueError("posterior_files must be a K:path map.")
    c["posterior_files"] = {str(k): resolve(v) for k, v in c["posterior_files"].items()}
    c["source_index_files"] = {
        str(k): resolve(v) for k, v in c.get("source_index_files", {}).items()
    }
    Ks = [integer(k, "candidate K") for k in c["candidate_Ks"]]
    if not Ks or len(set(Ks)) != len(Ks) or any(k not in range(4, 10) for k in Ks):
        raise ValueError("candidate_Ks must be distinct values in 4..9.")
    c["candidate_Ks"] = sorted(Ks)
    if set(c["posterior_files"]) != set(map(str, Ks)):
        raise ValueError("posterior_files must contain exactly all candidate_Ks.")
    if len(set(c["posterior_files"].values())) != len(Ks):
        raise ValueError("Each K must use its OWN distinct posterior file.")
    if set(c["source_index_files"]) - set(map(str, Ks)):
        raise ValueError("Unexpected source_index_files K.")
    for key in (
        "n_folds",
        "fold_seed",
        "iterations",
        "burnin",
        "score_thin",
        "trace_thin",
        "mcmc_seed",
    ):
        c[key] = integer(c[key], key)
    if (
        not 2 <= c["n_folds"] <= 32767
        or not 0 <= c["burnin"] < c["iterations"]
        or min(c["score_thin"], c["trace_thin"]) < 1
    ):
        raise ValueError("Invalid fold count, iterations/burnin, or thinning.")
    if any(not 0 <= c[key] < 2**32 for key in ("fold_seed", "mcmc_seed")):
        raise ValueError("Invalid seed.")
    if c["fold_stratification"] not in ("donor", "donor_cell_type"):
        raise ValueError("Invalid fold_stratification.")
    if not math.isfinite(float(c["z_offset"])):
        raise ValueError("Invalid z_offset.")
    for key in ("assume_same_cell_order", "save_scoring_draws"):
        if not isinstance(c[key], bool):
            raise ValueError(f"{key} must be true or false.")
    return c


def main():
    args = parse_args()
    config = read_config(args.config)
    dat = load_application_data(config)
    out = config["output_dir"]
    Ks = config["candidate_Ks"]
    F = config["n_folds"]
    execution_Ks = Ks if args.Ks is None else args.Ks
    execution_folds = list(range(1, F + 1)) if args.folds is None else args.folds
    if (
        not execution_Ks
        or len(set(execution_Ks)) != len(execution_Ks)
        or not set(execution_Ks) <= set(Ks)
    ):
        raise ValueError("--Ks must be a distinct subset of candidate_Ks.")
    if (
        not execution_folds
        or len(set(execution_folds)) != len(execution_folds)
        or not set(execution_folds) <= set(range(1, F + 1))
    ):
        raise ValueError("--folds must be a distinct subset of configured folds.")
    for p in [
        dat.path,
        *config["posterior_files"].values(),
        *config["source_index_files"].values(),
        args.config.resolve(),
    ]:
        if p.is_relative_to(out):
            raise ValueError("Output cannot contain/overwrite input files.")
    for p in config["posterior_files"].values():
        if out.is_relative_to(p.parent):
            raise ValueError("Output must be separate from full-fit directories.")
    times = {}
    sources = {}
    for K in Ks:
        times[K], sources[K] = load_source(K, config["posterior_files"][str(K)], dat, config)
    hp = sources[Ks[0]]["hyperparameters"]
    if any(s["hyperparameters"] != hp for s in sources.values()):
        raise ValueError("Prior hyperparameters differ between candidate full fits.")
    assignments = make_folds(dat, config)
    fold_hash = vector_hash(assignments, dtype="<i2")
    frozen = {
        "method": METHOD,
        "version": VERSION,
        "configuration": jsonable(config),
        "implementation_sha256": file_hash(Path(__file__).resolve()),
        "sampler_sha256": file_hash(mcmc.__file__),
        "cv_helpers_sha256": file_hash(conditional_cv.__file__),
        "data_sha256": dat.data_sha256,
        "data_identity_sha256": dat.identity_sha256,
        "fold_assignment_sha256": fold_hash,
        "fixed_T_sources": sources,
        "primary_rule": {"scope": "all", "weighting": "donor_balanced", "component": "lpd_xy"},
    }
    fingerprint = object_hash(frozen)
    owner = out / "experiment.json"
    if out.exists() and any(out.iterdir()):
        if not owner.is_file() or read_json(owner).get("fingerprint") != fingerprint:
            raise ValueError(
                "output_dir is not owned by this exact configuration. Use a NEW output_dir for changed inputs/settings."
            )
    jobs = {}
    for K in Ks:
        for fold in range(1, F + 1):
            seed = int(np.random.SeedSequence([config["mcmc_seed"], K, fold]).generate_state(1)[0])
            settings = {
                name: config[name] for name in ("iterations", "burnin", "score_thin", "trace_thin")
            }
            settings.update(seed=seed, hyperparameters=hp)
            job = {
                "method": METHOD,
                "version": VERSION,
                "run_fingerprint": fingerprint,
                "K": K,
                "fold": fold,
                "n_folds": F,
                "settings": settings,
                "fixed_time_source": sources[K],
                "fold_assignment_sha256": fold_hash,
                "save_scoring_draws": config["save_scoring_draws"],
            }
            jobs[K, fold] = (job, out / f"K_{K}" / f"fold_{fold}")
    expected_S = (config["iterations"] - config["burnin"] + config["score_thin"] - 1) // config[
        "score_thin"
    ]
    print(
        f"Validated {len(dat.counts)} donors, {dat.counts.sum()} cells; candidates={Ks}, folds={F}."
    )
    print(f"z_offset={config['z_offset']}; {expected_S} scoring draws/fold; priors={hp}")
    print("Alignment: " + ", ".join(sorted({s["alignment"] for s in sources.values()})))
    print(
        f"Requested execution: {len(execution_Ks) * len(execution_folds)} fold fits; T is never updated/integrated."
    )
    if args.check_only:
        print("Checks passed. No results written or MCMC fits run.")
        return
    if args.summarize_only and not owner.exists():
        raise FileNotFoundError("No existing experiment to summarize.")
    if not args.summarize_only:
        for K in execution_Ks:
            for fold in execution_folds:
                job, folder = jobs[K, fold]
                validate_destination(args, folder, job)
    out.mkdir(parents=True, exist_ok=True)
    lock = out / "_RUNNING.lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FileExistsError(
            f"{lock}: another run may be active. After confirming it has stopped, remove this lock to restart."
        ) from exc
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(f"pid={os.getpid()}\n")
        if not owner.exists():
            save_json(owner, {"method": METHOD, "fingerprint": fingerprint})
        save_json(
            out / "run_config.json",
            {**frozen, "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__},
        )
        save_json(out / "fixed_T_sources.json", sources)
        save_csv(
            out / "fold_assignments.csv",
            (
                {
                    "donor_index": i,
                    "donor_id": dat.donor_ids[i],
                    "cell_index": c,
                    "cell_id": dat.cell_ids[i][c],
                    "cell_type": int(dat.state[i][c]),
                    "fold": int(f) + 1,
                }
                for i, row in enumerate(assignments)
                for c, f in enumerate(row)
            ),
        )
        for K in Ks:
            path = out / "reference" / f"K_{K}" / "T_fixed_full.json"
            if not path.exists():
                save_json(path, times[K])
        if not args.summarize_only:
            # Remove the previous selection before reruns so a failed rerun cannot leave a stale decision.
            for name in (
                "selection.json",
                "cv_summary_by_K.csv",
                "donor_scores_by_K.csv",
                "fold_scores.csv",
                "cell_lpd_all.csv",
            ):
                (out / name).unlink(missing_ok=True)
            for K in execution_Ks:
                for fold in execution_folds:
                    job, folder = jobs[K, fold]
                    if not args.overwrite and completed_job(folder, job):
                        print(f"Skip completed K={K}, fold={fold}", flush=True)
                        continue
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / "_SUCCESS.json").unlink(missing_ok=True)
                    (folder / "posterior_draws.npz").unlink(missing_ok=True)
                    save_json(folder / "task_request.json", job)
                    fit_one(dat, times[K], assignments, job, folder)
        summarize(config, dat, assignments, times, jobs)
        print(f"Saved results: {out}")
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, ImportError, FloatingPointError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
