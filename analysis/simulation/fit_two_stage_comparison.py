"""Two-stage comparisons on the paired finite-sample simulation at K=5 and n=1000."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import invgamma

from blafm import mcmc

VERSION = "2.0"
K, N, DEGREE = 5, 1000, 3
KNOTS = np.linspace(0.0, 1.0, K - DEGREE + 1)
PRIOR = dict(ig_shape=2.0, ig_scale=0.1, gamma_shape=10.0, gamma_rate=0.1)
METHODS = ("joint", "twostage_t", "twostage_coef", "oracle_t")
STAGE1_FOLDER = "stage1_hierarchical"
DEFAULT_OUTPUT = "outputs/two_stage"
STAGE1_TRACE_KEYS = ("mu_alpha", "mu_beta", "v_alpha", "v_beta")


@dataclass(frozen=True)
class Stage1Prior:
    """Independent hyperpriors: mu ~ N(0, mu_sd^2), v ~ IG(shape, scale).

    The mean prior is independent of v (not N(0, c*v)). IG uses density
    proportional to v**(-shape-1) * exp(-scale/v). Defaults are working
    choices, not fitted using simulation truth or the Joint posterior.
    """

    mu_alpha_sd: float = 10.0
    mu_beta_sd: float = 10.0
    var_alpha_shape: float = 2.0
    var_alpha_scale: float = 0.1
    var_beta_shape: float = 2.0
    var_beta_scale: float = 0.1

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in asdict(self).values()):
            raise ValueError("Stage 1 hyperprior parameters must be finite and positive")


def stage1_prior_from_args(args):
    return Stage1Prior(
        **{name: getattr(args, "stage1_" + name) for name in Stage1Prior.__dataclass_fields__}
    )


def stage1_prior_description(prior):
    return {
        "model": "independent_hierarchical_normal_diagonal",
        "hyperparameters": asdict(prior),
        "population_mean_prior_center": 0.0,
        "population_mean_prior_independent_of_variance": True,
        "variance_parameterization": "inverse_gamma_shape_scale",
        "variances_shared_across_donors": True,
        "uses_Z_or_mediation_regression": False,
    }


def read_json(path):
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temp, path)


def atomic_npz(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, **obj)
    os.replace(temp, path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_array(value, name, shape=None):
    a = np.asarray(value, dtype=float)
    if shape is not None and a.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: nonfinite values")
    return a


class Data:
    def __init__(self, directory, expected_n=N):
        self.directory = Path(directory)
        xr = read_json(self.directory / "X.json")
        yr = read_json(self.directory / "Y.json")
        sr = read_json(self.directory / "cell_type.json")
        self.n = len(xr)
        if self.n != expected_n or len(yr) != self.n or len(sr) != self.n:
            raise ValueError(f"Expected {expected_n} donors in X, Y and cell_type")
        self.counts = np.array([len(x) for x in xr], dtype=int)
        if np.any(self.counts < 1):
            raise ValueError("Every donor needs at least one cell")
        self.cmax = int(self.counts.max())
        self.mask = np.arange(self.cmax)[None, :] < self.counts[:, None]
        self.X = self.pack(xr, "X")
        self.Y = self.pack(yr, "Y")
        self.Z = finite_array(read_json(self.directory / "Z.json"), "Z").reshape(-1)
        if self.Z.shape != (self.n,):
            raise ValueError("Z must have one scalar per donor")
        spad = self.pack(sr, "cell_type")
        if not np.all(np.isin(spad[self.mask], [0, 1])):
            raise ValueError("cell_type must contain only 0 and 1")
        self.S = spad.astype(np.int8)
        self.S[~self.mask] = -1
        self.state_ragged = [self.S[i, :c].tolist() for i, c in enumerate(self.counts)]
        self.low = np.where(self.S == 0, 0.0, 0.5)
        self.high = np.where(self.S == 0, 0.5, 1.0)
        self.observed_hashes = {
            f: sha256(self.directory / f) for f in ("X.json", "Y.json", "Z.json", "cell_type.json")
        }
        # Fixed diagnostic coordinates; never selected using truth or fit quality.
        flat_valid = np.flatnonzero(self.mask.ravel())
        positions = np.linspace(0, len(flat_valid) - 1, min(12, len(flat_valid)), dtype=int)
        self.probe_flat = flat_valid[positions]
        self.probe_donors = np.unique(self.probe_flat // self.cmax)

    def pack(self, rows, name):
        if len(rows) != self.n:
            raise ValueError(f"{name}: wrong donor count")
        out = np.zeros((self.n, self.cmax))
        for i, c in enumerate(self.counts):
            out[i, :c] = finite_array(rows[i], f"{name}[{i}]", (int(c),))
        return out

    def check_t(self, t, name):
        t = finite_array(t, name, self.X.shape)
        if np.any(t[self.mask] < self.low[self.mask] - 1e-12) or np.any(
            t[self.mask] > self.high[self.mask] + 1e-12
        ):
            raise ValueError(f"{name} violates the cell-state support")
        if np.any(t[~self.mask] != 0):
            raise ValueError(f"{name}: padded cells must be zero")
        return t.copy()

    def true_t(self):
        # Called only for the oracle fit (and separately by evaluation).
        return self.check_t(self.pack(read_json(self.directory / "T.json"), "T truth"), "T truth")


def make_phi(data, t):
    phi = mcmc.compute_bspline_basis_ragged(
        [t[i, :c] for i, c in enumerate(data.counts)], KNOTS, DEGREE
    )
    phi = np.asarray([mcmc.pad_phi(p, data.cmax) for p in phi])
    gram = np.einsum("nck,ncl->nkl", phi, phi)
    return phi, gram


def hierarchical_coefficients(data, phi, gram, sigma2, tau2, mu_alpha, v_alpha, mu_beta, v_beta):
    """Gaussian Gibbs draws conditional on t and the current population layer.

    Q = Phi'Phi / observation_variance + diag(1/v)
    q = Phi'X / observation_variance + mu/v. No Z, A, B or C enters.
    """
    draws = []
    for obs, var, mu, v in ((data.X, sigma2, mu_alpha, v_alpha), (data.Y, tau2, mu_beta, v_beta)):
        mu = finite_array(mu, "population mean", (K,))
        v = finite_array(v, "population variance", (K,))
        if np.any(v <= 0):
            raise ValueError("Population variances must be positive")
        score = np.einsum("nc,nck->nk", obs, phi) / var[:, None] + mu / v
        Q = gram / var[:, None, None] + np.diag(1.0 / v)[None, :, :]
        L = np.linalg.cholesky(Q)
        mean = np.linalg.solve(Q, score[..., None])[..., 0]
        noise = np.linalg.solve(np.swapaxes(L, -1, -2), np.random.normal(size=(data.n, K, 1)))[
            ..., 0
        ]
        draws.append(mean + noise)
    return draws


def update_population_mean(coefficients, variance, prior_sd):
    """mu | coefficients,v ~ N(V*sum(coeff)/v, V), V=(n/v+1/prior_sd^2)^-1."""
    n = len(coefficients)
    posterior_var = 1.0 / (n / variance + 1.0 / prior_sd**2)
    posterior_mean = posterior_var * coefficients.sum(axis=0) / variance
    return posterior_mean + np.sqrt(posterior_var) * np.random.normal(size=posterior_mean.shape)


def update_population_variance(coefficients, mean, shape, scale):
    """v | coefficients,mu ~ IG(a+n/2, b+sum((coeff-mu)^2)/2).

    No extra 1/2 or mu^2 term: p(mu) is independent of v. The original
    FMA update_Sigma/update_Lambda functions therefore cannot be reused here.
    """
    posterior_shape = shape + len(coefficients) / 2.0
    posterior_scale = scale + 0.5 * np.sum((coefficients - mean) ** 2, axis=0)
    return np.asarray(invgamma.rvs(a=posterior_shape, scale=posterior_scale))


def effect_coefficients(A, b, C):
    """Compute within each draw BEFORE posterior averaging."""
    return {"ME": A * b, "DE": C, "TE": A * b + C}


def basis_integrals():
    return mcmc.cubic_basis(K).integrate(0.0, 1.0)


def chain_ess(x):
    """Single-chain raw-scale ESS using paired initial-positive autocorrelations.

    A diagnostic only: not rank-normalized bulk ESS, not R-hat or convergence proof.
    """
    x = np.asarray(x, float)
    n = len(x)
    if n < 8 or not np.all(np.isfinite(x)) or np.var(x) <= 0:
        return None
    x = x - x.mean()
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, size)
    ac = np.fft.irfft(f * f.conjugate(), size)[:n]
    ac = ac / ac[0]
    pairs = ac[: 2 * (n // 2)].reshape(-1, 2).sum(axis=1)
    positive = []
    last = float("inf")
    for v in pairs:
        if v <= 0:
            break
        last = min(last, float(v))
        positive.append(last)
    if not positive:
        return float(n)
    iact = max(1.0, -1.0 + 2.0 * sum(positive))
    return float(min(n, n / iact))


def trace_diagnostics(arrays):
    diag = {}
    for name, raw in arrays.items():
        a = np.asarray(raw)
        if a.ndim == 1:
            a = a[:, None]
        for j in range(a.shape[1]):
            x = a[:, j]
            ess = chain_ess(x)
            diag[f"{name}[{j}]"] = {
                "mean": float(x.mean()),
                "sd": float(x.std(ddof=1)) if len(x) > 1 else None,
                "ess_raw_single_chain": ess,
                "mcmc_mcse_mean": float(x.std(ddof=1) / np.sqrt(ess)) if ess else None,
                "first_half_mean": float(x[: max(1, len(x) // 2)].mean()),
                "second_half_mean": float(x[len(x) // 2 :].mean()),
            }
    return diag


def progress(total, label, quiet=False):
    if quiet:
        return range(total)
    try:
        from tqdm import trange

        return trange(total, desc=label, dynamic_ncols=True)
    except ImportError:

        def fallback():
            start, last = time.monotonic(), 0.0
            for i in range(total):
                now = time.monotonic()
                if now - last >= 10 or i == total - 1:
                    elapsed = now - start
                    eta = elapsed * (total - i) / i if i else 0
                    print(
                        f"{label}: {i}/{total}, elapsed={elapsed:.0f}s, ETA={eta:.0f}s", flush=True
                    )
                    last = now
                yield i

        return fallback()


def run_mcmc(
    data, method, iterations, burnin, seed, *, stage1=None, quiet=False, stage1_prior=None
):
    if method not in ("stage1", *METHODS):
        raise ValueError(method)
    if not 0 <= burnin < iterations or iterations - burnin < 2:
        raise ValueError("Require 0 <= burnin < iterations and >=2 retained draws")
    prior1 = Stage1Prior() if stage1_prior is None else stage1_prior
    if not isinstance(prior1, Stage1Prior):
        raise TypeError("stage1_prior must be a Stage1Prior instance")
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    latent_t = method in ("stage1", "joint")
    if method == "oracle_t":
        t = data.true_t()
    elif method in ("twostage_t", "twostage_coef"):
        if stage1 is None:
            raise ValueError("Two-stage methods require hierarchical Stage 1")
        t = data.check_t(stage1["T_mean"], "Stage 1 T_mean")
    else:
        t = np.where(data.mask, data.low + (data.high - data.low) * rng.random(data.X.shape), 0.0)
    initial_t = t.copy()
    phi, gram = make_phi(data, t)

    alpha0, beta0 = np.zeros((1, K)), np.zeros((1, K))
    A, C, B = np.full((1, K), 0.1), np.full((1, K), 0.1), np.eye(K) * 0.1
    if method == "stage1":
        alpha, beta = np.zeros((data.n, K)), np.zeros((data.n, K))
        mu_alpha, mu_beta = np.zeros(K), np.zeros(K)
        v_alpha, v_beta = np.ones(K), np.ones(K)
    elif method in ("twostage_t", "twostage_coef"):
        alpha = finite_array(stage1["alpha_mean"], "Stage 1 alpha", (data.n, K)).copy()
        beta = finite_array(stage1["beta_mean"], "Stage 1 beta", (data.n, K)).copy()
    else:
        alpha = alpha0 + data.Z[:, None] * A
        beta = beta0 + data.Z[:, None] * C + alpha @ B.T
    Sigma, Lambda = np.eye(K), np.eye(K)
    sigma2, tau2 = np.full(data.n, 0.1), np.full(data.n, 0.1)
    if method == "twostage_t":
        sigma2 = stage1["sigma2_mean"].copy()
        tau2 = stage1["tau2_mean"].copy()
    tau_a, tau_d = 10.0, 10.0
    kept = iterations - burnin
    sums = {key: np.zeros_like(x) for key, x in (("T", t), ("alpha", alpha), ("beta", beta))}
    sums2 = {k: np.zeros_like(v) for k, v in sums.items()}
    noise_sums = {k: np.zeros(data.n) for k in ("sigma2", "tau2")}
    accepted = np.zeros_like(t)
    traces = {
        "t_probe": np.empty((kept, len(data.probe_flat))),
        "alpha_probe": np.empty((kept, len(data.probe_donors))),
        "beta_probe": np.empty((kept, len(data.probe_donors))),
    }
    if method == "stage1":
        for key in STAGE1_TRACE_KEYS:
            traces[key] = np.empty((kept, K))
    else:
        for key, dim in (
            ("A", K),
            ("b", K),
            ("C", K),
            ("alpha0", K),
            ("beta0", K),
            ("Sigma_diag", K),
            ("Lambda_diag", K),
            ("tau_a", 1),
            ("tau_d", 1),
        ):
            traces[key] = np.empty((kept, dim))
    for itr in progress(iterations, method, quiet):
        if latent_t:
            t, phi, gram, _, _, _, accept = mcmc.update_T_A_fast(
                data.X,
                data.Y,
                alpha,
                beta,
                phi,
                t,
                sigma2,
                tau2,
                data.n,
                data.cmax,
                data.counts,
                KNOTS,
                K,
                data.state_ragged,
                sigma=1.0,
                rng=rng,
                use_q=True,
                cell_type_pad=data.S,
                return_phi=True,
            )
        if method == "stage1":
            alpha, beta = hierarchical_coefficients(
                data, phi, gram, sigma2, tau2, mu_alpha, v_alpha, mu_beta, v_beta
            )
            mu_alpha = update_population_mean(alpha, v_alpha, prior1.mu_alpha_sd)
            mu_beta = update_population_mean(beta, v_beta, prior1.mu_beta_sd)
            v_alpha = update_population_variance(
                alpha, mu_alpha, prior1.var_alpha_shape, prior1.var_alpha_scale
            )
            v_beta = update_population_variance(
                beta, mu_beta, prior1.var_beta_shape, prior1.var_beta_scale
            )
        elif method != "twostage_coef":
            alpha, beta = mcmc.update_alpha_beta_block(
                alpha0,
                A,
                B,
                beta0,
                C,
                data.X,
                data.Y,
                data.Z,
                phi,
                gram,
                Sigma,
                Lambda,
                sigma2,
                tau2,
                data.n,
            )
        if method != "twostage_coef":
            sigma2 = np.asarray(
                mcmc.update_sigma_i(
                    data.X, data.n, data.counts, alpha, phi, PRIOR["ig_shape"], PRIOR["ig_scale"]
                )
            )
            tau2 = np.asarray(
                mcmc.update_tau_i(
                    data.Y, data.n, data.counts, beta, phi, PRIOR["ig_shape"], PRIOR["ig_scale"]
                )
            )
        if method != "stage1":
            alpha0, A = mcmc.update_alpha0_A(alpha, data.Z, Sigma, data.n, K, tau_a)
            theta = mcmc.update_beta0_BC_diag(data.Z, beta, alpha, Lambda, data.n, K, tau_d)
            beta0, C, B = theta[:, 0][None, :], theta[:, 1][None, :], np.diag(theta[:, 2])
            Sigma = mcmc.update_Sigma(
                alpha, alpha0, A, data.Z, data.n, K, tau_a, PRIOR["ig_shape"], PRIOR["ig_scale"]
            )
            Lambda = mcmc.update_Lambda(
                alpha,
                beta,
                beta0,
                B,
                C,
                data.Z,
                data.n,
                K,
                tau_d,
                PRIOR["ig_shape"],
                PRIOR["ig_scale"],
            )
            tau_a = float(
                mcmc.update_tau_a(
                    PRIOR["gamma_shape"], PRIOR["gamma_rate"], alpha0, A, Sigma, K, rng=rng
                )
            )
            tau_d = float(
                mcmc.update_tau_d(
                    PRIOR["gamma_shape"], PRIOR["gamma_rate"], beta0, C, B, Lambda, K, rng=rng
                )
            )
        if itr >= burnin:
            j = itr - burnin
            for key, val in (("T", t), ("alpha", alpha), ("beta", beta)):
                sums[key] += val
                sums2[key] += val * val
            if method != "twostage_coef":
                noise_sums["sigma2"] += sigma2
                noise_sums["tau2"] += tau2
            if latent_t:
                accepted += accept
            traces["t_probe"][j] = t.ravel()[data.probe_flat]
            traces["alpha_probe"][j] = alpha[data.probe_donors, 0]
            traces["beta_probe"][j] = beta[data.probe_donors, 0]
            if method == "stage1":
                for key, val in (
                    ("mu_alpha", mu_alpha),
                    ("mu_beta", mu_beta),
                    ("v_alpha", v_alpha),
                    ("v_beta", v_beta),
                ):
                    traces[key][j] = val
            else:
                for key, val in (
                    ("A", A.ravel()),
                    ("b", np.diag(B)),
                    ("C", C.ravel()),
                    ("alpha0", alpha0.ravel()),
                    ("beta0", beta0.ravel()),
                    ("Sigma_diag", np.diag(Sigma)),
                    ("Lambda_diag", np.diag(Lambda)),
                    ("tau_a", [tau_a]),
                    ("tau_d", [tau_d]),
                ):
                    traces[key][j] = val
    out = dict(traces)
    if method == "stage1":
        for key in STAGE1_TRACE_KEYS:
            out[key + "_mean"] = traces[key].mean(axis=0)
    for key in sums:
        out[key + "_mean"] = sums[key] / kept
        out[key + "_var"] = np.maximum(0.0, sums2[key] / kept - out[key + "_mean"] ** 2)
    if not latent_t:
        # Avoid accumulated rounding error for quantities that are fixed by design.
        out["T_mean"], out["T_var"] = initial_t, np.zeros_like(t)
    if method == "twostage_coef":
        for key in ("alpha", "beta"):
            out[key + "_mean"] = stage1[key + "_mean"].copy()
            out[key + "_var"] = np.zeros((data.n, K))
    else:
        for key, value in noise_sums.items():
            out[key + "_mean"] = value / kept
    out["cell_counts"] = data.counts
    out["probe_flat_indices"] = data.probe_flat
    out["probe_donor_indices"] = data.probe_donors
    out["saved_iterations"] = np.arange(burnin + 1, iterations + 1)
    if latent_t:
        out["T_acceptance_postburn"] = accepted / kept
    elapsed = time.perf_counter() - start
    diagnostic_traces = {k: v for k, v in traces.items() if k in ("A", "b", "C")}
    if method == "stage1":
        diagnostic_traces = traces
    else:
        for effect, coeff in effect_coefficients(out["A"], out["b"], out["C"]).items():
            diagnostic_traces["I" + effect] = coeff @ basis_integrals()
    diag = {
        "wall_seconds": elapsed,
        "retained_draws": kept,
        "T_updated": latent_t,
        "coefficients_updated": method != "twostage_coef",
        "stage1_population_parameters_updated": method == "stage1",
        "T_acceptance_mean": float(np.mean(accepted[data.mask] / kept)) if latent_t else None,
        "trace_diagnostics": trace_diagnostics(diagnostic_traces),
        "note": "Single-chain diagnostics are not evidence of multi-chain convergence. "
        "Replicates use different datasets and must not be combined for R-hat.",
    }
    for key, val in out.items():
        if not np.all(np.isfinite(val)):
            raise FloatingPointError(f"{method}: nonfinite output in {key}")
    return out, diag


def import_legacy_joint(data, replicate):
    folder = data.directory.parent / "fit_K5"
    config_path, posterior_path = folder / "fit_config.json", folder / "posterior_set.json"
    cfg = read_json(config_path)
    for key, value in (("K", K), ("n", N), ("replicate", replicate)):
        if cfg.get(key) != value:
            raise ValueError(f"{config_path}: {key} should equal {value}")
    iterations, burnin = int(cfg["iterations"]), int(cfg["burnin"])
    if not 0 <= burnin < iterations - 1:
        raise ValueError("Invalid burn-in in legacy fit_config.json")
    p = read_json(posterior_path)
    out = {}
    # Original JSON includes the initial state at index 0. Never burn twice.
    for key, target in (
        ("A", "A"),
        ("B", "b"),
        ("C", "C"),
        ("alpha0", "alpha0"),
        ("beta0", "beta0"),
    ):
        chain = finite_array(p[key], f"legacy {key}")
        if len(chain) != iterations + 1:
            raise ValueError(
                f"Legacy {key} must contain iterations+1 states; "
                "thinned/already-trimmed chains need an explicit importer"
            )
        chain = chain[burnin + 1 :]
        if key == "B":
            if chain.shape[1:] != (K, K):
                raise ValueError("Legacy B shape mismatch")
            diag = np.diagonal(chain, axis1=1, axis2=2).copy()
            if not np.allclose(chain, diag[:, :, None] * np.eye(K), atol=1e-12, rtol=0):
                raise ValueError("Legacy B must be diagonal")
            out[target] = diag
        else:
            out[target] = chain.reshape(len(chain), K)
    out["T_mean"] = data.check_t(p["T_pred"], "legacy T_pred")
    for key in ("alpha", "beta"):
        out[key + "_mean"] = finite_array(p[key + "_pred"], key, (N, K))
    out["cell_counts"] = data.counts
    out["saved_iterations"] = np.arange(burnin + 1, iterations + 1)
    dtrace = {k: out[k] for k in ("A", "b", "C")}
    for effect, coeff in effect_coefficients(out["A"], out["b"], out["C"]).items():
        dtrace["I" + effect] = coeff @ basis_integrals()
    info = {
        "wall_seconds": None,
        "retained_draws": iterations - burnin,
        "legacy_config": cfg,
        "trace_diagnostics": trace_diagnostics(dtrace),
        "legacy_source": str(posterior_path.resolve()),
        "legacy_posterior_sha256": sha256(posterior_path),
        "legacy_config_sha256": sha256(config_path),
        "note": "Imported post-burn-in population draws and donor means from the paired joint fit.",
    }
    return out, info


def base_request(data, args, method):
    method_ids = {"joint": 1, "stage1": 2, "twostage_t": 3, "twostage_coef": 4, "oracle_t": 5}
    seed = int(
        np.random.SeedSequence([args.seed, args.replicate, method_ids[method]]).generate_state(1)[0]
    )
    return {
        "version": VERSION,
        "method": method,
        "replicate": args.replicate,
        "K": K,
        "n": N,
        "data_directory": str(data.directory.resolve()),
        "observed_sha256": data.observed_hashes,
        "iterations": args.iterations,
        "burnin": args.burnin,
        "seed": seed,
        "priors": PRIOR,
        "stage1_prior": stage1_prior_description(stage1_prior_from_args(args)),
        "knots": KNOTS.tolist(),
        "degree": DEGREE,
        "mh_proposal_sd": 1.0,
        "driver_sha256": sha256(__file__),
        "sampler_sha256": sha256(mcmc.__file__),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }


def get_or_fit(folder, request, producer, overwrite=False):
    folder = Path(folder)
    result_path, config_path = folder / "posterior.npz", folder / "fit_config.json"
    if result_path.exists() and config_path.exists() and not overwrite:
        old = read_json(config_path)
        if old.get("request") != request:
            raise RuntimeError(
                f"{folder}: saved configuration differs. Use a different "
                "output root or --overwrite; results are not silently reused."
            )
        if old.get("result_sha256") != sha256(result_path):
            raise RuntimeError(f"{folder}: result checksum mismatch")
        print(f"Reuse completed stage: {folder}", flush=True)
        with np.load(result_path, allow_pickle=False) as f:
            return {k: f[k] for k in f.files}
    if (result_path.exists() or config_path.exists()) and not overwrite:
        raise RuntimeError(
            f"{folder}: incomplete output. Rerun with --overwrite "
            "or remove only this incomplete method folder."
        )
    print(f"Run: {request['method']}, replicate={request['replicate']}", flush=True)
    arrays, diagnostics = producer()
    atomic_npz(result_path, arrays)
    atomic_json(folder / "diagnostics.json", diagnostics)
    atomic_json(
        config_path,
        {
            "request": request,
            "diagnostics": diagnostics,
            "result_sha256": sha256(result_path),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    return arrays


def validate_input_files(data, use_oracle=True):
    # Only dimensions/metadata here; truth values never enter non-oracle fitting.
    path_truth = read_json(data.directory / "path_truth.json")
    if path_truth.get("basis_dimension") != K or path_truth.get("spline_degree") != DEGREE:
        raise ValueError("This comparison requires K_true=5 and cubic splines")
    if use_oracle:
        data.true_t()
    for f in ("effect_truth.json", "grid.json", "alpha.json", "beta.json"):
        if not (data.directory / f).is_file():
            raise FileNotFoundError(data.directory / f)


def run_replicate(args):
    root, output = args.root.resolve(), args.output_root.resolve()
    if root == output or root in output.parents:
        raise ValueError("output-root must be outside the original data root")
    directory = root / f"replicate_{args.replicate:03d}" / "n_1000" / "generated_data"
    data = Data(directory)
    validate_input_files(data)
    if args.joint_source == "reuse":
        old = directory.parent / "fit_K5"
        for name in ("posterior_set.json", "fit_config.json"):
            if not (old / name).is_file():
                raise FileNotFoundError(
                    f"{old / name}; provide the original Joint fit, or use --joint-source refit"
                )
        old_cfg = read_json(old / "fit_config.json")
        for key, val in (("K", K), ("n", N), ("replicate", args.replicate)):
            if old_cfg.get(key) != val:
                raise ValueError(f"{old / 'fit_config.json'}: wrong {key}")
        recorded = old_cfg.get("data_sha256")
        if recorded is not None and any(
            recorded.get(Path(name).stem) != val for name, val in data.observed_hashes.items()
        ):
            raise ValueError("The joint fit and stage-one inputs refer to different observations.")
        if old_cfg["iterations"] != args.iterations or old_cfg["burnin"] != args.burnin:
            raise ValueError("The paired joint fit uses different iteration or burn-in settings.")
        if old_cfg.get("posterior_sha256") and old_cfg["posterior_sha256"] != sha256(
            old / "posterior_set.json"
        ):
            raise ValueError(
                "The saved joint posterior checksum differs from its fit configuration."
            )
    destination = output / f"replicate_{args.replicate:03d}"
    if args.check_only:
        print(f"Input OK: rep={args.replicate}, donors={N}, cells={data.counts.sum()}, K={K}")
        return
    # Both two-stage methods use this exact same hierarchical Stage 1 fit.
    req1 = base_request(data, args, "stage1")
    stage1 = get_or_fit(
        destination / STAGE1_FOLDER,
        req1,
        lambda: run_mcmc(
            data,
            "stage1",
            args.iterations,
            args.burnin,
            req1["seed"],
            stage1_prior=stage1_prior_from_args(args),
            quiet=args.quiet,
        ),
        args.overwrite,
    )
    parent_hash = sha256(destination / STAGE1_FOLDER / "posterior.npz")
    for method in METHODS:
        req = base_request(data, args, method)
        if method.startswith("twostage"):
            req["stage1_result_sha256"] = parent_hash
        if method == "oracle_t":
            req["true_t_sha256"] = sha256(directory / "T.json")
        if method == "joint" and args.joint_source == "reuse":
            old = directory.parent / "fit_K5"
            # Fail clearly if the expected legacy baseline is absent.
            req["source"] = "legacy_joint_reuse"
            req["legacy_posterior_sha256"] = sha256(old / "posterior_set.json")
            req["legacy_config_sha256"] = sha256(old / "fit_config.json")
            producer = lambda: import_legacy_joint(data, args.replicate)
        else:
            req["source"] = "new_mcmc"
            producer = lambda method=method, req=req: run_mcmc(
                data,
                method,
                args.iterations,
                args.burnin,
                req["seed"],
                stage1=stage1,
                quiet=args.quiet,
            )
        get_or_fit(destination / method, req, producer, args.overwrite)
    print(f"Completed replicate {args.replicate:03d}: {destination}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/finite"))
    parser.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=10000, help="TOTAL, including burn-in")
    parser.add_argument("--burnin", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261107)
    for name, default in asdict(Stage1Prior()).items():
        parser.add_argument("--stage1-" + name.replace("_", "-"), type=float, default=default)
    parser.add_argument("--joint-source", choices=("reuse", "refit"), default="reuse")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace comparison outputs only; original fit_K5 is never overwritten",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.replicate < 1 or args.seed < 0:
        parser.error("replicate >= 1 and seed >= 0 required")
    if not 0 <= args.burnin < args.iterations - 1:
        parser.error("Require 0 <= burnin < iterations-1")
    try:
        stage1_prior_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


if __name__ == "__main__":
    run_replicate(parse_args())
