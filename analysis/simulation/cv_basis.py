"""Donor-balanced conditional cross-validation with each candidate's full-data state estimates."""

from __future__ import annotations
from blafm.mcmc import (
    compute_bspline_basis_ragged,
    pad_phi,
    update_Lambda,
    update_Sigma,
    update_alpha0_A,
    update_alpha_beta_block,
    update_beta0_BC_diag,
    update_sigma_i,
    update_tau_a,
    update_tau_d,
    update_tau_i,
)
from analysis.conditional_cv import (
    build_fixed_test_basis,
    file_hash,
    integer,
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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

# Respect existing thread settings; avoid oversubscription for small matrices.
for _thread_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_name, "1")

import numpy as np
import scipy

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(values, **kwargs):
        print(kwargs.get("desc", "MCMC"), flush=True)
        return values


VERSION = "1.0.0"
METHOD = "scheme2_fixed_full_data_own_K_T_pred_conditional"
SCORE_TARGET = "conditional_fixed_own_K_T_pred"
LOG_2PI = math.log(2.0 * math.pi)
HP_KEYS = ("a1", "b1", "a2", "b2", "a3", "b3", "a4", "b4", "a_a", "b_a", "a_b", "b_b")
SCOPES = ("all", "early", "late")
COMPONENTS = ("lpd_x", "lpd_y_given_x", "lpd_xy")


# Original data splitting; conditional fixed-time predictive density scoring.


def write_cell_lpd(
    path: Path,
    heldout: dict[str, np.ndarray],
    K: int,
    fold_one_based: int,
    n_score_draws: int,
    lpd_x: np.ndarray,
    lpd_xy: np.ndarray,
) -> None:
    lpd_y_given_x = lpd_xy - lpd_x
    fieldnames = [
        "K",
        "fold",
        "donor_index",
        "donor_id",
        "cell_index",
        "cell_id",
        "cell_type",
        "X",
        "Y",
        "lpd_x",
        "lpd_y_given_x",
        "lpd_xy",
        "n_score_draws",
        "t_hat",
        "time_source_K",
    ]

    order = np.lexsort((heldout["cell_index"], heldout["donor_index"]))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row_index in order:
            donor_index = int(heldout["donor_index"][row_index])
            cell_index = int(heldout["cell_index"][row_index])
            writer.writerow(
                {
                    "K": K,
                    "fold": fold_one_based,
                    "donor_index": donor_index,
                    "donor_id": donor_index + 1,
                    "cell_index": cell_index,
                    "cell_id": cell_index + 1,
                    "cell_type": int(heldout["state"][row_index]),
                    "X": float(heldout["x"][row_index]),
                    "Y": float(heldout["y"][row_index]),
                    "lpd_x": float(lpd_x[row_index]),
                    "lpd_y_given_x": float(lpd_y_given_x[row_index]),
                    "lpd_xy": float(lpd_xy[row_index]),
                    "n_score_draws": n_score_draws,
                    "t_hat": float(heldout["t_hat"][row_index]),
                    "time_source_K": K,
                }
            )


# Experiment-specific input handling and execution.


def replicate_number(value):
    return integer(str(value).removeprefix("replicate_"), "replicate")


def formatted_path(root, template, rep, K, fold=1):
    path = Path(template.format(replicate=f"replicate_{rep:03d}", rep=rep, K=K, fold=fold))
    return path if path.is_absolute() else root / path


def load_dataset(root, rep):
    folder = root / f"replicate_{rep:03d}" / "generated_data"
    arrays = {}
    for name in ("X", "Y", "cell_type"):
        raw = read_json(folder / f"{name}.json")
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"{folder}/{name}.json: expected nested donor/cell arrays.")
        rows = [np.asarray(row, dtype=float) for row in raw]
        if any(row.ndim != 1 or len(row) == 0 or not np.isfinite(row).all() for row in rows):
            raise ValueError(f"{folder}/{name}.json: empty, nonfinite or nonvector donor row.")
        arrays[name] = rows
    counts = np.asarray([len(row) for row in arrays["X"]], dtype=int)
    if any([len(row) for row in arrays[name]] != counts.tolist() for name in ("Y", "cell_type")):
        raise ValueError(f"{folder}: X, Y and cell_type shapes differ.")
    if any(not np.isin(row, [0, 1]).all() for row in arrays["cell_type"]):
        raise ValueError(f"{folder}: state must be 0=early or 1=late.")
    if np.any(counts < 2):
        raise ValueError(f"{folder}: each donor needs at least two cells for cell CV.")
    z = np.asarray(read_json(folder / "Z.json"), dtype=float).reshape(-1)
    if len(z) != len(counts) or not np.isfinite(z).all():
        raise ValueError(f"{folder}: invalid Z or donor count mismatch.")
    return SimpleNamespace(
        rep=rep,
        folder=folder,
        X=arrays["X"],
        Y=arrays["Y"],
        Z=z,
        state=[row.astype(np.int8) for row in arrays["cell_type"]],
        counts=counts,
        hashes={name: file_hash(folder / f"{name}.json") for name in ("X", "Y", "Z", "cell_type")},
    )


def load_fixed_times(path, key, dat, source_K):
    print(f"Reading K={source_K} {key}: {path}", flush=True)
    try:
        import ijson
    except ImportError:
        print(
            "  ijson unavailable: loading whole posterior JSON (pip install ijson reduces memory).",
            flush=True,
        )
        doc = read_json(path)
        if key not in doc:
            raise KeyError(
                f"{path}: missing exact key {key!r}; no automatic time-key substitution."
            )
        raw = doc[key]
    else:
        missing = object()
        with path.open("rb") as handle:
            raw = next(ijson.items(handle, key, use_float=True), missing)
        if raw is missing:
            raise KeyError(
                f"{path}: missing exact key {key!r}; no automatic time-key substitution."
            )
    if not isinstance(raw, list) or len(raw) != len(dat.counts):
        raise ValueError(f"{path}[{key!r}]: expected one row per original donor.")
    rows = [np.asarray(row, dtype=float) for row in raw]
    if any(row.ndim != 1 for row in rows):
        raise ValueError(f"{key}: expected cell estimates, not an array of MCMC draws.")
    widths = np.asarray([len(row) for row in rows])
    if not (np.array_equal(widths, dat.counts) or np.all(widths == dat.counts.max())):
        raise ValueError(
            f"{key}: row lengths must match actual counts or padding to their maximum."
        )
    output = []
    for i, (row, n, state) in enumerate(zip(rows, dat.counts, dat.state)):
        values = row[:n].copy()
        lo = 0.5 * state
        if not np.isfinite(values).all() or np.any(values < lo) or np.any(values > lo + 0.5):
            raise ValueError(
                f"{key}: invalid time or state interval at donor_index={i}; check original ordering."
            )
        values.flags.writeable = False
        output.append(values)
    config_path = path.parent / "fit_config.json"
    cfg_hash = None
    if config_path.exists():
        cfg = read_json(config_path)
        if integer(cfg.get("K"), "time source K") != source_K or "fold" in cfg:
            raise ValueError(
                f"{config_path}: expected the full-data K={source_K} fit for this candidate."
            )
        if "replicate" in cfg and replicate_number(cfg["replicate"]) != dat.rep:
            raise ValueError(f"{config_path}: time source belongs to another replicate.")
        for name, expected in dat.hashes.items():
            if name in cfg.get("data_sha256", {}) and cfg["data_sha256"][name] != expected:
                raise ValueError(f"{config_path}: time-source data hash differs for {name}.")
        cfg_hash = file_hash(config_path)
        recorded = cfg.get("data")
        if isinstance(recorded, str):
            candidates = [Path(recorded), dat.folder.parents[2] / recorded]
            accessible = next((d for d in candidates if (d / "X.json").exists()), None)
            if accessible is not None:
                for name, expected in dat.hashes.items():
                    if (
                        not (accessible / f"{name}.json").is_file()
                        or file_hash(accessible / f"{name}.json") != expected
                    ):
                        raise ValueError(f"{config_path}: recorded data differ for {name}.")
    return output, {
        "path": str(path.resolve()),
        "key": key,
        "trimmed_times_sha256": vector_hash(output),
        "file_size": path.stat().st_size,
        "fit_config_sha256": cfg_hash,
        "source_fit": f"full_data_K{source_K}",
        "source_K": source_K,
        "reference_strategy": "own_candidate_K",
        "reference_uses_heldout_expression": True,
    }


def read_fold_assignments(args, datasets):
    # Use the same donor/state-stratified folds for every candidate K.
    cfg = read_json(args.paper_config)["basis"]
    assignments = {}
    for rep, dat in datasets.items():
        rng = np.random.default_rng(cfg["fold_seed"])
        rows = []
        for state in dat.state:
            folds = np.full(len(state), -1, dtype=np.int16)
            for group in (0, 1):
                ind = np.flatnonzero(state == group)
                if not len(ind):
                    continue
                shuffled = ind[rng.permutation(len(ind))]
                labels = np.arange(len(ind)) % args.n_folds
                labels = labels[rng.permutation(len(ind))]
                folds[shuffled] = labels
            if np.any(folds < 0):
                raise ValueError("Incomplete fold assignment.")
            rows.append(folds)
        if set(np.concatenate(rows)) != set(range(args.n_folds)):
            raise ValueError("At least one CV fold is globally empty.")
        assignments[rep] = rows
    return assignments


def read_settings(args, dat, K, fold, assignment_hash):
    cfg = read_json(args.paper_config)
    sim, basis = cfg["simulation"], cfg["basis"]
    defaults = dict(
        burnin=sim["burnin"],
        iterations=sim["iterations"],
        score_thin=basis["score_thin"],
        trace_thin=basis["trace_thin"],
    )
    values = {
        k: integer(getattr(args, k) if getattr(args, k) is not None else v, k)
        for k, v in defaults.items()
    }
    seed_base = args.seed if args.seed is not None else basis["cv_mcmc_seed_base"]
    values["seed"] = integer(seed_base + dat.rep * 1000 + K * 10 + fold, "seed")
    if (
        not 0 <= values["burnin"] < values["iterations"]
        or min(values["score_thin"], values["trace_thin"]) < 1
    ):
        raise ValueError("Invalid burnin, iterations or thinning.")
    hp = {k: float(cfg["hyperparameters"][k]) for k in HP_KEYS}
    if any(not math.isfinite(v) or v <= 0 for v in hp.values()):
        raise ValueError("Invalid prior.")
    values.update(
        hyperparameters=hp,
        original_config_path=str(args.paper_config.resolve()),
        original_config_sha256=file_hash(args.paper_config),
        fold_assignment_sha256=assignment_hash,
    )
    return values


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--paper-config", type=Path, default=Path(__file__).resolve().parents[1] / "settings.json"
    )
    p.add_argument("--root", type=Path, default=Path("outputs/basis"))
    p.add_argument("--replicates", nargs="+", type=int, default=list(range(1, 11)))
    p.add_argument("--Ks", nargs="+", type=int, choices=range(4, 10), default=list(range(4, 10)))
    p.add_argument("--folds", nargs="+", type=int)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--time-template", default="{replicate}/fit_K_{K}/posterior_set.json")
    p.add_argument(
        "--time-file", type=Path, help="Explicit posterior for ONE replicate and ONE candidate K."
    )
    p.add_argument("--time-key", default="T_pred")
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--save-draws",
        action="store_true",
        help="Save post-burn-in scoring parameter draws to posterior_draws.npz.",
    )
    p.add_argument(
        "--selection-weighting",
        choices=("donor_balanced", "cell_weighted"),
        default="donor_balanced",
    )
    p.add_argument("--selection-component", choices=COMPONENTS, default="lpd_xy")
    for name in ("burnin", "iterations", "score-thin", "trace-thin", "seed"):
        p.add_argument(
            "--" + name,
            type=int,
            default=None,
            help="Override manuscript CV settings for a new run.",
        )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--restart-incomplete", action="store_true")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun selected fits and replace previous script results in --output.",
    )
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Validate all selected inputs without fitting or writing results.",
    )
    args = p.parse_args()
    if args.overwrite and (args.resume or args.restart_incomplete):
        p.error("--overwrite cannot be combined with --resume or --restart-incomplete.")
    if args.n_folds < 2 or args.n_folds > 32767:
        p.error("n-folds must be between 2 and 32767.")
    for name in ("replicates", "Ks"):
        v = getattr(args, name)
        if not v or len(v) != len(set(v)) or min(v) < 1:
            p.error(f"{name} must contain distinct positive integers.")
    if args.folds is None:
        args.folds = list(range(1, args.n_folds + 1))
    if len(args.folds) != len(set(args.folds)) or any(
        f < 1 or f > args.n_folds for f in args.folds
    ):
        p.error("folds must be distinct and within 1..n-folds.")
    if args.time_file is not None and (len(args.replicates) != 1 or len(args.Ks) != 1):
        p.error("--time-file requires exactly one replicate and one candidate K.")
    if not args.time_key or "." in args.time_key:
        p.error("Invalid reference-K or time-key (must be one top-level JSON key).")
    args.root = args.root.resolve()
    args.output = (args.output or args.root / "cell_cv_scheme2_own_T_pred").resolve()
    if args.output == args.root:
        p.error("Output must be a separate experiment directory, not the data root.")
    return args


def fit_one(dat, fixed_times, assignment, job, folder):
    K, fold = job["K"], job["fold"]
    if job["fixed_time_source"].get("source_K") != K:
        raise ValueError("Scheme 2 requires each candidate to use its own full-fit time estimates.")
    if vector_hash(fixed_times) != job["fixed_time_source"]["trimmed_times_sha256"]:
        raise ValueError("Fixed times differ from the validated candidate-specific source.")
    settings = SimpleNamespace(**job["settings"])
    hp = settings.hyperparameters
    X_train, Y_train, states, heldout = split_one_fold(
        dat.X, dat.Y, dat.state, assignment, fold - 1
    )
    indices = [np.flatnonzero(row != fold - 1) for row in assignment]
    T = [row[ind].copy() for row, ind in zip(fixed_times, indices)]
    for row in T:
        row.flags.writeable = False
    fixed_before = vector_hash(T)
    n = len(T)
    C_d = np.asarray([len(row) for row in T], dtype=int)
    c_max = int(C_d.max())
    X_pad = np.asarray([np.pad(row, (0, c_max - len(row))) for row in X_train])
    Y_pad = np.asarray([np.pad(row, (0, c_max - len(row))) for row in Y_train])
    knots = np.linspace(0.0, 1.0, K - 2)
    phi = np.asarray(
        [pad_phi(row, c_max) for row in compute_bspline_basis_ragged(T, knots, degree=3)]
    )
    phi_phi_t = np.einsum("ikj,ijh->ikh", np.transpose(phi, (0, 2, 1)), phi)
    # Read-only arrays make accidental in-place time/design updates an error.
    phi.flags.writeable = False
    phi_phi_t.flags.writeable = False
    np.random.seed(settings.seed)
    rng = np.random.default_rng(settings.seed)
    alpha0 = np.zeros((1, K))
    A = np.full((1, K), 0.1)
    beta0 = np.zeros((1, K))
    C = np.full((1, K), 0.1)
    B = np.diagflat(np.full(K, 0.1))
    alpha = alpha0 + A * dat.Z.reshape(-1, 1)
    beta = beta0 + C * dat.Z.reshape(-1, 1) + alpha @ B.T
    Sigma = np.eye(K)
    Lambda = np.eye(K)
    sigma_i = np.full(n, 0.1)
    tau_i = np.full(n, 0.1)
    tau_a = 10.0
    tau_d = 10.0
    sums = {
        name: np.zeros_like(value)
        for name, value in {
            "alpha0": alpha0,
            "A": A,
            "beta0": beta0,
            "B": B,
            "C": C,
            "Sigma": Sigma,
            "Lambda": Lambda,
            "sigma_i": sigma_i,
            "tau_i": tau_i,
            "alpha": alpha,
            "beta": beta,
        }.items()
    }
    sums["tau_a"] = 0.0
    sums["tau_d"] = 0.0
    phi_test = build_fixed_test_basis(heldout, fixed_times, knots, degree=3)
    acc_x = np.full(len(heldout["x"]), -np.inf)
    acc_xy = np.full(len(heldout["x"]), -np.inf)
    trace = []
    n_retained = 0
    n_score = 0
    saved_draws = None
    if job["save_scoring_draws"]:
        S = (settings.iterations - settings.burnin + settings.score_thin - 1) // settings.score_thin
        saved_draws = {
            "alpha": np.empty((S, n, K)),
            "beta": np.empty((S, n, K)),
            "sigma_i": np.empty((S, n)),
            "tau_i": np.empty((S, n)),
            "iteration": np.empty(S, dtype=np.int64),
        }
    started = time.monotonic()
    for itr in tqdm(
        range(settings.iterations),
        desc=f"rep={dat.rep} K={K} fold={fold} fixed T_pred / conditional score",
    ):
        # NO update_T call, time proposals, jitter, rank remapping, or time randomization.
        alpha, beta = update_alpha_beta_block(
            alpha0,
            A,
            B,
            beta0,
            C,
            X_pad,
            Y_pad,
            dat.Z,
            phi,
            phi_phi_t,
            Sigma,
            Lambda,
            sigma_i,
            tau_i,
            n,
        )
        sigma_i = np.asarray(update_sigma_i(X_pad, n, C_d, alpha, phi, hp["a1"], hp["b1"]))
        tau_i = np.asarray(update_tau_i(Y_pad, n, C_d, beta, phi, hp["a2"], hp["b2"]))
        alpha0, A = update_alpha0_A(alpha, dat.Z, Sigma, n, K, tau_a)
        theta = update_beta0_BC_diag(dat.Z, beta, alpha, Lambda, n, K, tau_d)
        beta0 = theta[:, 0].reshape(1, -1)
        C = theta[:, 1].reshape(1, -1)
        B = np.diag(theta[:, 2])
        Sigma = update_Sigma(alpha, alpha0, A, dat.Z, n, K, tau_a, hp["a3"], hp["b3"])
        Lambda = update_Lambda(alpha, beta, beta0, B, C, dat.Z, n, K, tau_d, hp["a4"], hp["b4"])
        tau_a = update_tau_a(hp["a_a"], hp["b_a"], alpha0, A, Sigma, K, rng=rng)
        tau_d = update_tau_d(hp["a_b"], hp["b_b"], beta0, C, B, Lambda, K, rng=rng)
        values = {
            "alpha0": alpha0,
            "A": A,
            "beta0": beta0,
            "B": B,
            "C": C,
            "Sigma": Sigma,
            "Lambda": Lambda,
            "sigma_i": sigma_i,
            "tau_i": tau_i,
            "alpha": alpha,
            "beta": beta,
            "tau_a": tau_a,
            "tau_d": tau_d,
        }
        if itr % settings.trace_thin == 0 or itr == settings.iterations - 1:
            row = {"iteration": itr + 1}
            row.update(
                {
                    name + "_mean": float(np.mean(value))
                    for name, value in values.items()
                    if name in ("alpha0", "A", "beta0", "C", "sigma_i", "tau_i")
                }
            )
            row.update(
                diagB_mean=float(np.mean(np.diag(B))),
                Sigma_diag_mean=float(np.mean(np.diag(Sigma))),
                Lambda_diag_mean=float(np.mean(np.diag(Lambda))),
                tau_a=float(tau_a),
                tau_d=float(tau_d),
                t00=float(T[0][0]),
            )
            trace.append(row)
        if itr >= settings.burnin:
            n_retained += 1
            for name, value in values.items():
                sums[name] += value
            if (itr - settings.burnin) % settings.score_thin == 0:
                update_conditional_log_accumulators(
                    heldout, phi_test, alpha, beta, sigma_i, tau_i, acc_x, acc_xy
                )
                if saved_draws is not None:
                    for name in ("alpha", "beta", "sigma_i", "tau_i"):
                        saved_draws[name][n_score] = values[name]
                    saved_draws["iteration"][n_score] = itr + 1
                n_score += 1
    if n_retained != settings.iterations - settings.burnin or n_score < 1:
        raise RuntimeError("Unexpected retained/scoring draw count.")
    if vector_hash(T) != fixed_before:
        raise RuntimeError("Training times changed during fixed-time fitting.")
    lpd_x = acc_x - math.log(n_score)
    lpd_xy = acc_xy - math.log(n_score)
    means = {name: value / n_retained for name, value in sums.items()}
    if (
        not np.isfinite(lpd_x).all()
        or not np.isfinite(lpd_xy).all()
        or any(not np.isfinite(v).all() for v in means.values())
    ):
        raise FloatingPointError("Nonfinite posterior mean or predictive score.")
    summary = summarize_fold_scores(heldout, lpd_x, lpd_xy)
    summary.update(
        replicate=dat.rep,
        K=K,
        fold=fold,
        n_folds=job["n_folds"],
        time_source_K=K,
        reference_strategy="own_candidate_K",
        n_train_cells=int(C_d.sum()),
        n_score_draws=n_score,
        n_retained_draws=n_retained,
        test_time_target=SCORE_TARGET,
        test_time_integrated=False,
        score_thin=settings.score_thin,
        fold_assignment_sha256=job["fold_assignment_sha256"],
        training_time_mode=METHOD,
        fixed_training_times_sha256=fixed_before,
        pseudotime_update_count=0,
        pseudotime_acceptance_rate=None,
        max_abs_training_time_change=0.0,
        elapsed_seconds=time.monotonic() - started,
    )
    write_cell_lpd(folder / "cell_lpd.csv", heldout, K, fold, n_score, lpd_x, lpd_xy)
    write_trace(folder / "trace.csv", trace)
    save_json(folder / "posterior_summary.json", means)
    save_json(folder / "fold_summary.json", summary)
    config = dict(job)
    config.update(
        data=str(dat.folder),
        output=str(folder),
        fit_mode="fixed-cv",
        time_source_K=K,
        reference_strategy="own_candidate_K",
        **{k: v for k, v in vars(settings).items() if k not in ("seed",)},
        mcmc_seed=settings.seed,
        fixed_training_times_sha256=fixed_before,
        test_time_target=SCORE_TARGET,
        test_T_pred_used=True,
        test_time_integrated=False,
        training_times_updated=False,
        reference_uses_heldout_expression=True,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    save_json(folder / "cv_config.json", config)
    files = (
        "cell_lpd.csv",
        "trace.csv",
        "posterior_summary.json",
        "fold_summary.json",
        "cv_config.json",
    )
    if saved_draws is not None:
        if len(saved_draws["iteration"]) != n_score:
            raise RuntimeError("Saved scoring draw count mismatch.")
        temporary = folder / "posterior_draws.npz.tmp"
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **saved_draws)
        temporary.replace(folder / "posterior_draws.npz")
        files += ("posterior_draws.npz",)
    save_json(
        folder / "_SUCCESS.json",
        {
            "method": METHOD,
            "version": VERSION,
            "task_fingerprint": object_hash(job),
            "output_sha256": {name: file_hash(folder / name) for name in files},
        },
    )
    print(
        f"  mean lpd_xy={summary['mean_cell_lpd_xy']:.6f}; donor-balanced={summary['donor_balanced_mean_cell_lpd_xy']:.6f}",
        flush=True,
    )
    return summary


def completed_job(folder, job, allow_incompatible=False):
    marker = folder / "_SUCCESS.json"
    if not marker.exists():
        return False
    saved = read_json(marker)
    if saved.get("method") != METHOD or saved.get("task_fingerprint") != object_hash(job):
        if allow_incompatible:
            print(f"Excluding incompatible unselected fit from summaries: {folder}", flush=True)
            return False
        raise ValueError(
            f"{folder}: completed fit settings/inputs differ; use --overwrite to rerun it or a separate output directory."
        )
    required = (
        "cell_lpd.csv",
        "trace.csv",
        "posterior_summary.json",
        "fold_summary.json",
        "cv_config.json",
    )
    if job["save_scoring_draws"]:
        required += ("posterior_draws.npz",)
    for name in required:
        expected = saved.get("output_sha256", {}).get(name)
        if (
            expected is None
            or not (folder / name).is_file()
            or file_hash(folder / name) != expected
        ):
            raise ValueError(f"{folder}: completed output missing or changed: {name}.")
    return True


def validate_job_destination(args, job, folder):
    """Check all selected destinations before any result is replaced."""
    if args.overwrite:
        if folder.exists() and any(folder.iterdir()):
            request = folder / "task_request.json"
            if not request.is_file() or read_json(request).get("method") not in (METHOD,):
                raise FileExistsError(
                    f"{folder}: not a fit created by this script; refusing to overwrite."
                )
            marker = folder / "_SUCCESS.json"
            if marker.exists() and read_json(marker).get("method") not in (METHOD,):
                raise FileExistsError(f"{folder}: completion marker belongs to another experiment.")
        return
    if completed_job(folder, job):
        if not args.resume:
            raise FileExistsError(
                f"{folder}: already complete; use --resume to skip it or --overwrite to rerun it."
            )
        return
    if folder.exists() and any(folder.iterdir()):
        request = folder / "task_request.json"
        if not request.exists() or read_json(request).get("method") != METHOD:
            raise FileExistsError(f"{folder}: not an incomplete fit created by this script.")
        if object_hash(read_json(request)) != object_hash(job):
            raise ValueError(
                f"{folder}: incomplete fit uses different settings/inputs; use --overwrite or another output."
            )
        if not args.restart_incomplete:
            raise FileExistsError(
                f"{folder}: incomplete fit; --restart-incomplete restarts this chain."
            )


def execute_job(args, dat, times, assignments, job, folder):
    validate_job_destination(args, job, folder)
    if not args.overwrite and completed_job(folder, job):
        print(f"Skipping completed rep={dat.rep}, K={job['K']}, fold={job['fold']}", flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / "_SUCCESS.json"
    if args.overwrite and marker.exists():
        # A failed replacement must never retain the old completion marker.
        marker.unlink()
    # A restarted/replaced chain must not leave stale draws from an older fit.
    (folder / "posterior_draws.npz").unlink(missing_ok=True)
    (folder / "posterior_draws.npz.tmp").unlink(missing_ok=True)
    save_json(folder / "task_request.json", job)
    fit_one(dat, times, assignments, job, folder)


def write_selections(args, scores):
    """Only select among complete, paired scores for EVERY requested K."""
    details = []
    expected = set(args.Ks)
    for rep in args.replicates:
        for weighting in ("cell_weighted", "donor_balanced"):
            candidates = [
                row
                for row in scores
                if row["replicate"] == rep
                and row["scope"] == "all"
                and row["weighting"] == weighting
            ]
            if {row["K"] for row in candidates} != expected:
                continue
            for component in COMPONENTS:
                ordered = sorted(candidates, key=lambda r: (-r[component], r["K"]))
                best = ordered[0]
                tied = [row["K"] for row in ordered if row[component] == best[component]]
                details.append(
                    dict(
                        replicate=rep,
                        target=SCORE_TARGET,
                        scope="all",
                        weighting=weighting,
                        component=component,
                        selected_K=best["K"],
                        best_lpd=best[component],
                        exactly_tied_Ks=";".join(map(str, sorted(tied))),
                        runner_up_K=ordered[1]["K"] if len(ordered) > 1 else None,
                        gap_to_runner_up=best[component] - ordered[1][component]
                        if len(ordered) > 1
                        else None,
                        candidate_Ks=";".join(map(str, sorted(expected))),
                        n_folds=args.n_folds,
                    )
                )
    fields = [
        "replicate",
        "target",
        "scope",
        "weighting",
        "component",
        "selected_K",
        "best_lpd",
        "exactly_tied_Ks",
        "runner_up_K",
        "gap_to_runner_up",
        "candidate_Ks",
        "n_folds",
    ]
    save_csv(args.output / "selection_details.csv", details, fields)
    primary = [
        row
        for row in details
        if row["weighting"] == args.selection_weighting
        and row["component"] == args.selection_component
    ]
    save_csv(args.output / "selection_by_replicate.csv", primary, fields)
    frequency = []
    for K in sorted(args.Ks):
        count = sum(row["selected_K"] == K for row in primary)
        frequency.append(
            dict(
                K=K,
                target=SCORE_TARGET,
                scope="all",
                weighting=args.selection_weighting,
                component=args.selection_component,
                n_selected=count,
                n_complete_replicates=len(primary),
                proportion=count / len(primary) if primary else None,
            )
        )
    save_csv(args.output / "selection_frequency.csv", frequency)
    if primary:
        print(
            f"Primary selections ({args.selection_weighting}, {args.selection_component}, all cells):",
            flush=True,
        )
        for row in primary:
            print(
                f"  replicate {row['replicate']}: K={row['selected_K']}, LPD={row['best_lpd']:.6f}",
                flush=True,
            )
    else:
        print("No K selection yet: every requested K must have all folds complete.", flush=True)
    return len(primary)


def summarize_results(args, datasets, assignments, jobs):
    out = args.output
    fold_rows = []
    scores = []
    all_csv = out / "cell_lpd_all.csv"
    temp_csv = out / "cell_lpd_all.csv.tmp"
    raw_fields = [
        "K",
        "fold",
        "donor_index",
        "donor_id",
        "cell_index",
        "cell_id",
        "cell_type",
        "X",
        "Y",
        "lpd_x",
        "lpd_y_given_x",
        "lpd_xy",
        "n_score_draws",
        "t_hat",
        "time_source_K",
    ]
    with temp_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["replicate", "training_time_mode", "target"] + raw_fields
        )
        writer.writeheader()
        for rep, dat in datasets.items():
            offsets = np.r_[0, np.cumsum(dat.counts)[:-1]]
            donor_ids = np.repeat(np.arange(len(dat.counts)), dat.counts)
            states = np.concatenate(dat.state)
            for K in args.Ks:
                seen = np.zeros(int(dat.counts.sum()), dtype=bool)
                lpd = np.full((len(seen), 3), np.nan)
                finished = []
                for fold in range(1, args.n_folds + 1):
                    job, folder = jobs[(rep, K, fold)]
                    if not completed_job(folder, job, allow_incompatible=fold not in args.folds):
                        continue
                    summary = read_json(folder / "fold_summary.json")
                    fold_rows.append(summary)
                    finished.append(fold)
                    count = 0
                    with (folder / "cell_lpd.csv").open(encoding="utf-8-sig", newline="") as source:
                        for row in csv.DictReader(source):
                            i, c = (
                                integer(row["donor_index"], "donor"),
                                integer(row["cell_index"], "cell"),
                            )
                            if not (0 <= i < len(dat.counts) and 0 <= c < dat.counts[i]):
                                raise ValueError(f"{folder}: invalid saved cell index.")
                            index = offsets[i] + c
                            if seen[index] or assignments[rep][i][c] != fold - 1:
                                raise ValueError(f"{folder}: duplicate cell or wrong saved fold.")
                            seen[index] = True
                            count += 1
                            lpd[index] = [float(row[name]) for name in COMPONENTS]
                            writer.writerow(
                                {
                                    "replicate": f"replicate_{rep:03d}",
                                    "training_time_mode": METHOD,
                                    "target": SCORE_TARGET,
                                    **row,
                                }
                            )
                    if count != summary["n_heldout_cells"]:
                        raise ValueError(f"{folder}: saved held-out cell count differs.")
                if len(finished) != args.n_folds:
                    continue
                if not seen.all() or not np.isfinite(lpd).all():
                    raise ValueError(f"replicate {rep}, K={K}: incomplete or nonfinite pooled CV.")
                for scope in SCOPES:
                    at = (
                        np.ones(len(seen), dtype=bool)
                        if scope == "all"
                        else states == int(scope == "late")
                    )
                    if not at.any():
                        continue
                    ids = donor_ids[at]
                    counts = np.bincount(ids, minlength=len(dat.counts))
                    present = counts > 0
                    for weighting in ("cell_weighted", "donor_balanced"):
                        row = {
                            "replicate": rep,
                            "K": K,
                            "target": SCORE_TARGET,
                            "scope": scope,
                            "weighting": weighting,
                            "n_cells": int(at.sum()),
                            "n_donors": int(present.sum()),
                            "n_folds": args.n_folds,
                        }
                        for j, name in enumerate(COMPONENTS):
                            v = lpd[at, j]
                            value = (
                                v.mean()
                                if weighting == "cell_weighted"
                                else np.mean(
                                    np.bincount(ids, weights=v, minlength=len(counts))[present]
                                    / counts[present]
                                )
                            )
                            row[name] = float(value)
                        scores.append(row)
    temp_csv.replace(all_csv)
    save_csv(
        out / "fold_scores.csv",
        fold_rows,
        fields=list(fold_rows[0]) if fold_rows else ["replicate", "K", "fold", "mean_cell_lpd_xy"],
    )
    score_fields = [
        "replicate",
        "K",
        "target",
        "scope",
        "weighting",
        "n_cells",
        "n_donors",
        "n_folds",
        *COMPONENTS,
    ]
    save_csv(out / "cv_scores_by_replicate.csv", scores, score_fields)

    def aggregate(rows, components, include_wins=False):
        grouped = defaultdict(list)
        for row in rows:
            grouped[(row["K"], row["scope"], row["weighting"])].append(row)
        results = []
        for (K, scope, weighting), group in sorted(grouped.items()):
            row = {
                "K": K,
                "target": SCORE_TARGET,
                "scope": scope,
                "weighting": weighting,
                "n_replicates": len(group),
            }
            for name in components:
                v = np.asarray([r[name] for r in group])
                row["mean_" + name] = float(v.mean())
                row["simulation_mcse_" + name] = (
                    float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None
                )
                if include_wins:
                    row["n_positive_" + name] = int((v > 0).sum())
            results.append(row)
        return results

    overall = aggregate(scores, COMPONENTS)
    save_csv(
        out / "overall_cv_summary.csv",
        overall,
        fields=list(overall[0])
        if overall
        else ["K", "scope", "weighting", "n_replicates", "mean_lpd_xy"],
    )
    lookup = {(row["replicate"], row["K"], row["scope"], row["weighting"]): row for row in scores}
    paired = []
    for row in scores:
        baseline = lookup.get((row["replicate"], 5, row["scope"], row["weighting"]))
        if row["K"] == 5 or baseline is None:
            continue
        if row["n_cells"] != baseline["n_cells"] or row["n_donors"] != baseline["n_donors"]:
            raise ValueError("Unpaired complete-CV comparison.")
        paired.append(
            {
                **{k: row[k] for k in ("replicate", "K", "target", "scope", "weighting")},
                "reference_K": 5,
                **{"delta_" + name: row[name] - baseline[name] for name in COMPONENTS},
            }
        )
    save_csv(
        out / "paired_K_minus_5.csv",
        paired,
        [
            "replicate",
            "K",
            "target",
            "scope",
            "weighting",
            "reference_K",
            *["delta_" + name for name in COMPONENTS],
        ],
    )
    pair_overall = aggregate(paired, ["delta_" + name for name in COMPONENTS], include_wins=True)
    save_csv(
        out / "overall_paired_K_minus_5.csv",
        pair_overall,
        fields=list(pair_overall[0])
        if pair_overall
        else ["K", "scope", "weighting", "n_replicates", "mean_delta_lpd_xy"],
    )
    if scores:
        print("Complete-CV mean joint LPD (larger is better):", flush=True)
        for row in overall:
            if row["scope"] == "all":
                print(
                    f"  K={row['K']} {row['weighting']}: {row['mean_lpd_xy']:.6f} (R={row['n_replicates']})",
                    flush=True,
                )
    else:
        print(
            "Partial folds only: see fold_scores.csv; no complete-CV replicate score yet.",
            flush=True,
        )
    n_selected = write_selections(args, scores)
    return {
        "n_complete_fits": len(fold_rows),
        "n_complete_replicate_K": len({(r["replicate"], r["K"]) for r in scores}),
        "n_complete_selections": n_selected,
    }


def main():
    args = parse_args()
    datasets = {rep: load_dataset(args.root, rep) for rep in args.replicates}
    assignments = read_fold_assignments(args, datasets)
    times = {}
    sources = {}
    jobs = {}
    source_paths = set()
    implementation_hash = file_hash(Path(__file__).resolve())
    print(
        "Scheme 2: each K uses its OWN full-data T_pred for training and held-out conditional LPD.",
        flush=True,
    )
    for rep, dat in datasets.items():
        fold_hash = vector_hash(assignments[rep], dtype="<i2")
        for K in args.Ks:
            path = (
                args.time_file or formatted_path(args.root, args.time_template, rep, K)
            ).resolve()
            if path in source_paths:
                raise ValueError(
                    "Scheme 2 requires distinct time files for each replicate/K; check --time-template."
                )
            source_paths.add(path)
            times[(rep, K)], sources[(rep, K)] = load_fixed_times(path, args.time_key, dat, K)
            for fold in range(1, args.n_folds + 1):
                settings = read_settings(args, dat, K, fold, fold_hash)
                train_indices = [np.flatnonzero(row != fold - 1) for row in assignments[rep]]
                if any(len(ind) == 0 for ind in train_indices):
                    raise ValueError(
                        f"replicate {rep}, fold {fold}: a donor has no training cells."
                    )
                job = {
                    "method": METHOD,
                    "version": VERSION,
                    "implementation_sha256": implementation_hash,
                    "sampler_sha256": file_hash(mcmc.__file__),
                    "cv_helpers_sha256": file_hash(conditional_cv.__file__),
                    "replicate": rep,
                    "K": K,
                    "fold": fold,
                    "n_folds": args.n_folds,
                    "data_sha256": dat.hashes,
                    "fold_assignment_sha256": fold_hash,
                    "fixed_time_source": sources[(rep, K)],
                    "save_scoring_draws": args.save_draws,
                    "reference_strategy": "own_candidate_K",
                    "settings": settings,
                }
                folder = args.output / f"replicate_{rep:03d}" / f"K_{K}" / f"fold_{fold}"
                jobs[(rep, K, fold)] = (job, folder)
        first = jobs[(rep, args.Ks[0], args.folds[0])][0]["settings"]
        print(
            f"Validated replicate {rep}: {len(dat.counts)} donors, {dat.counts.sum()} cells; "
            f"first task iterations={first['iterations']}, burnin={first['burnin']}, "
            f"score_thin={first['score_thin']}; test time fixed (no integration)",
            flush=True,
        )
        print("  Manuscript priors:", first["hyperparameters"], flush=True)
    # Do not write inside, or as an ancestor of, any original input directory.
    protected = [dat.folder.resolve() for dat in datasets.values()]
    protected += [Path(source["path"]).parent for source in sources.values()]
    protected += [
        Path(job["settings"]["original_config_path"]).parent for job, folder in jobs.values()
    ]
    for source_dir in protected:
        if args.output.is_relative_to(source_dir) or source_dir.is_relative_to(args.output):
            raise ValueError(
                "Output must not overlap original data, full fits or cell-CV fit directories."
            )
    if args.check_only:
        print("Input checks passed. No fitting or result writing was performed.", flush=True)
        return
    # Protect unrelated output directories, including all original CV results.
    owner = args.output / "experiment.json"
    if args.output.exists() and any(args.output.iterdir()):
        allowed_methods = (METHOD,)
        if not owner.exists() or read_json(owner).get("method") not in allowed_methods:
            raise FileExistsError(
                f"{args.output}: nonempty directory not owned by this experiment."
            )
    reference_configs = {}
    for rep, dat in datasets.items():
        for K in args.Ks:
            reference_configs[(rep, K)] = {
                "source": sources[(rep, K)],
                "data_sha256": dat.hashes,
                "reference_strategy": "own_candidate_K",
                "fold_assignment_sha256": vector_hash(assignments[rep], dtype="<i2"),
            }
            existing = (
                args.output
                / f"replicate_{rep:03d}"
                / "reference"
                / f"K_{K}"
                / "reference_config.json"
            )
            if (
                existing.exists()
                and object_hash(read_json(existing)) != object_hash(reference_configs[(rep, K)])
                and not args.overwrite
            ):
                raise ValueError(
                    f"{existing.parent}: existing reference inputs differ; use --overwrite or another output."
                )
        for fold in args.folds:
            for K in args.Ks:
                job, folder = jobs[(rep, K, fold)]
                validate_job_destination(args, job, folder)
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(
        owner,
        {
            "method": METHOD,
            "version": VERSION,
            "root": str(args.root),
            "reference_strategy": "own_candidate_K",
            "reference_uses_heldout_expression": True,
            "test_time_target": SCORE_TARGET,
            "test_time_integrated": False,
        },
    )
    success = args.output / "_SUCCESS.json"
    if success.exists():
        success.unlink()
    save_json(
        args.output / "run_config.json",
        {
            "arguments": vars(args),
            "method": METHOD,
            "version": VERSION,
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "implementation_sha256": implementation_hash,
            "sampler_sha256": file_hash(mcmc.__file__),
            "cv_helpers_sha256": file_hash(conditional_cv.__file__),
            "reference_uses_heldout_expression": True,
            "reference_strategy": "own_candidate_K",
            "true_T_read": False,
            "completed": False,
        },
    )
    for rep, dat in datasets.items():
        for K in args.Ks:
            reference_dir = args.output / f"replicate_{rep:03d}" / "reference" / f"K_{K}"
            save_json(reference_dir / "reference_config.json", reference_configs[(rep, K)])
            save_json(reference_dir / "T_fixed_full.json", times[(rep, K)])
            save_json(
                reference_dir / "fold_assignments.json", [row + 1 for row in assignments[rep]]
            )
        for fold in args.folds:
            ind = [np.flatnonzero(row != fold - 1) for row in assignments[rep]]
            for K in args.Ks:
                reference_dir = args.output / f"replicate_{rep:03d}" / "reference" / f"K_{K}"
                save_json(reference_dir / f"fold_{fold}" / "train_cell_indices.json", ind)
                save_json(
                    reference_dir / f"fold_{fold}" / "T_fixed_train.json",
                    [row[idx] for row, idx in zip(times[(rep, K)], ind)],
                )
                job, folder = jobs[(rep, K, fold)]
                execute_job(args, dat, times[(rep, K)], assignments[rep], job, folder)
    summary = summarize_results(args, datasets, assignments, jobs)
    save_json(
        args.output / "_SUCCESS.json",
        {
            "method": METHOD,
            "version": VERSION,
            **summary,
            "requested_replicates": args.replicates,
            "requested_Ks": args.Ks,
            "requested_folds": args.folds,
            "requested_fits_completed": True,
            "reference_uses_heldout_expression": True,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    run = read_json(args.output / "run_config.json")
    run["completed"] = True
    save_json(args.output / "run_config.json", run)
    print(f"Saved: {args.output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError, FileExistsError, FloatingPointError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
