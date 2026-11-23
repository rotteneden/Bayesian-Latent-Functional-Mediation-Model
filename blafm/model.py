"""Fit a joint latent functional mediation model to donor-grouped cells."""

from dataclasses import dataclass

import numpy as np
from tqdm.auto import trange

from . import mcmc


def _prepare(x, y, z, cell_type):
    n = len(x)
    if n == 0 or len(y) != n or len(cell_type) != n:
        raise ValueError("x, y and cell_type must have one row for each donor.")

    z = np.asarray(z, dtype=float)
    if z.shape == (n, 1):
        z = z[:, 0]
    if z.shape != (n,) or not np.isfinite(z).all():
        raise ValueError("z must contain one finite exposure per donor.")

    rows = []
    for i, group in enumerate(zip(x, y, cell_type)):
        dx, dy, state = (np.asarray(row, dtype=float) for row in group)
        if dx.ndim != 1 or dx.size == 0 or dx.shape != dy.shape or dx.shape != state.shape:
            raise ValueError(f"Donor {i} needs matching, nonempty vectors for x, y and cell_type.")
        if not np.isfinite(dx).all() or not np.isfinite(dy).all():
            raise ValueError(f"Donor {i} has nonfinite expression.")
        if not np.isin(state, (0, 1)).all():
            raise ValueError(f"Donor {i} has a cell_type outside 0 and 1.")
        rows.append((dx, dy, state))

    counts = np.array([len(dx) for dx, _, _ in rows])
    shape = (n, counts.max())
    x_pad, y_pad, state_pad = (np.zeros(shape) for _ in range(3))
    for i, (dx, dy, state) in enumerate(rows):
        x_pad[i, : counts[i]] = dx
        y_pad[i, : counts[i]] = dy
        state_pad[i, : counts[i]] = state
    return x_pad, y_pad, z, state_pad, counts


PAPER_PRIORS = {
    "a1": 2.0,
    "b1": 0.1,
    "a2": 2.0,
    "b2": 0.1,
    "a3": 2.0,
    "b3": 0.1,
    "a4": 2.0,
    "b4": 0.1,
    "a_a": 10.0,
    "b_a": 0.1,
    "a_b": 10.0,
    "b_b": 0.1,
}


@dataclass
class ChainState:
    """One MCMC state. The initial state has iteration zero."""

    times: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    population: tuple
    auxiliary: tuple
    accepted: np.ndarray


def sample_chain(
    x, y, z, cell_type, *, K, iterations, seed, progress=False, fixed_times=None, priors=None
):
    """Yield the initialization followed by one state per completed iteration."""
    if not isinstance(K, (int, np.integer)) or K < 4:
        raise ValueError("K must be an integer of at least 4.")
    if not isinstance(iterations, (int, np.integer)) or iterations < 2:
        raise ValueError("iterations must be an integer of at least 2.")
    if not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    hp = PAPER_PRIORS if priors is None else priors
    if set(hp) != set(PAPER_PRIORS) or any(not np.isfinite(v) or v <= 0 for v in hp.values()):
        raise ValueError(
            "Provide a positive finite value for each variance and multiplier hyperparameter."
        )

    xp, yp, z, state, counts = _prepare(x, y, z, cell_type)
    n, cmax = xp.shape
    valid = np.arange(cmax) < counts[:, None]
    lower = 0.5 * state[valid]
    knots = np.linspace(0, 1, K - 2)
    rng = np.random.default_rng(seed)
    random = np.random.RandomState(seed)
    times = np.zeros_like(xp)
    if fixed_times is None:
        times[valid] = lower + 0.5 * rng.random(counts.sum())
    else:
        if len(fixed_times) != n or any(
            np.shape(row) != (c,) for row, c in zip(fixed_times, counts)
        ):
            raise ValueError("Fixed states must match the observed cells within each donor.")
        fixed = np.asarray(np.concatenate(fixed_times), dtype=float)
        if not np.isfinite(fixed).all() or np.any(fixed < lower) or np.any(fixed > lower + 0.5):
            raise ValueError("Fixed states fall outside their annotation intervals.")
        times[valid] = fixed

    phi = np.zeros((*xp.shape, K))
    phi[valid] = mcmc.bspline_basis_matrix(times[valid], knots)
    gram = np.einsum("nci,ncj->nij", phi, phi)
    alpha0, beta0 = np.zeros((1, K)), np.zeros((1, K))
    A, C, B = np.full((1, K), 0.1), np.full((1, K), 0.1), np.eye(K) * 0.1
    alpha = alpha0 + z[:, None] * A
    beta = beta0 + z[:, None] * C + alpha @ B.T
    Sigma, Lambda = np.eye(K), np.eye(K)
    var_x, var_y = np.full((2, n), 0.1)
    tau_a = tau_d = 10.0
    accepted = np.zeros_like(valid)

    def snapshot():
        return ChainState(
            times,
            alpha,
            beta,
            (alpha0, A, beta0, B, C),
            (Sigma, Lambda, var_x, var_y, tau_a, tau_d),
            accepted,
        )

    yield snapshot()
    for _ in trange(iterations, disable=not progress, desc=f"BLaFM K={K}"):
        if fixed_times is None:
            times, phi, gram, _, _, _, accepted = mcmc.update_T_A_fast(
                xp,
                yp,
                alpha,
                beta,
                phi,
                times,
                var_x,
                var_y,
                n,
                cmax,
                counts,
                knots,
                K,
                cell_type,
                rng=rng,
                cell_type_pad=state,
            )
        alpha, beta = mcmc.update_alpha_beta_block(
            alpha0, A, B, beta0, C, xp, yp, z, phi, gram, Sigma, Lambda, var_x, var_y, n, random
        )
        var_x = mcmc.update_sigma_i(xp, n, counts, alpha, phi, hp["a1"], hp["b1"], random)
        var_y = mcmc.update_tau_i(yp, n, counts, beta, phi, hp["a2"], hp["b2"], random)
        alpha0, A = mcmc.update_alpha0_A(alpha, z, Sigma, n, K, tau_a, random)
        theta = mcmc.update_beta0_BC_diag(z, beta, alpha, Lambda, n, K, tau_d, random)
        beta0, C, B = theta[:, 0][None, :], theta[:, 1][None, :], np.diag(theta[:, 2])
        Sigma = mcmc.update_Sigma(alpha, alpha0, A, z, n, K, tau_a, hp["a3"], hp["b3"], random)
        Lambda = mcmc.update_Lambda(
            alpha, beta, beta0, B, C, z, n, K, tau_d, hp["a4"], hp["b4"], random
        )
        tau_a = mcmc.update_tau_a(hp["a_a"], hp["b_a"], alpha0, A, Sigma, K, rng=rng)
        tau_d = mcmc.update_tau_d(hp["a_b"], hp["b_b"], beta0, C, B, Lambda, K, rng=rng)
        yield snapshot()


def fit(x, y, z, cell_type, *, K=5, iterations=10_000, burnin=None, seed=123, progress=False):
    """Fit BLaFM and return post-burn-in draws and donor/cell posterior means.

    Each of x, y and cell_type contains one cell vector per donor. z contains
    one scalar exposure per donor. The time means follow input order and
    cell_counts gives the number of cells contributed by each donor.
    """
    if not isinstance(iterations, (int, np.integer)) or iterations < 2:
        raise ValueError("iterations must be an integer of at least 2.")
    burnin = iterations // 2 if burnin is None else burnin
    if not isinstance(burnin, (int, np.integer)) or not 0 <= burnin <= iterations - 2:
        raise ValueError("burnin must leave at least two posterior draws.")
    chain = sample_chain(
        x, y, z, cell_type, K=K, iterations=iterations, seed=seed, progress=progress
    )
    initial = next(chain)
    counts = np.asarray([len(row) for row in x])
    valid = np.arange(initial.times.shape[1]) < counts[:, None]
    retained = iterations - burnin
    names = ("alpha0", "A", "beta0", "B_diag", "C")
    draws = {key: np.empty((retained, K)) for key in names}
    t_sum = np.zeros_like(initial.times)
    alpha_sum = np.zeros_like(initial.alpha)
    beta_sum = np.zeros_like(initial.beta)
    for itr, current in enumerate(chain, start=1):
        if itr <= burnin:
            continue
        for name, arr in zip(names, current.population):
            draws[name][itr - burnin - 1] = np.diag(arr) if name == "B_diag" else arr.ravel()
        t_sum += current.times
        alpha_sum += current.alpha
        beta_sum += current.beta
    draws.update(
        t_mean=t_sum[valid] / retained,
        alpha_mean=alpha_sum / retained,
        beta_mean=beta_sum / retained,
        cell_counts=counts,
        K=K,
        seed=seed,
        iterations=iterations,
        burnin=burnin,
    )
    return draws
