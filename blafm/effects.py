"""Posterior summaries for mediated, direct and total effects."""

import numpy as np

from .mcmc import cubic_basis


def _summary(draws):
    lower, upper = np.quantile(draws, (0.025, 0.975), axis=0)
    return np.stack((draws.mean(axis=0), draws.std(axis=0, ddof=1), lower, upper), axis=-1)


def summarize_effects(fit, grid=None):
    """Summarize matched posterior draws for a unit increase in exposure.

    ME, DE and TE have one row per grid point. IME, IDE and ITE integrate
    uniformly on [0, 1] within each draw. Each output has four columns in the
    order given by summary_columns.
    """
    a, b, c = (np.asarray(fit[key], dtype=float) for key in ("A", "B_diag", "C"))
    if a.ndim != 2 or a.shape != b.shape or a.shape != c.shape:
        raise ValueError("A, B_diag and C must have matching (draw, basis) dimensions.")
    if a.shape[0] < 2 or a.shape[1] < 4 or not all(np.isfinite(p).all() for p in (a, b, c)):
        raise ValueError("Expected at least two finite posterior draws for a cubic basis.")

    if grid is None:
        grid = np.linspace(0.0, 1.0, 101)
    grid = np.asarray(grid, dtype=float)
    if (
        grid.ndim != 1
        or not grid.size
        or not np.isfinite(grid).all()
        or np.any((grid < 0) | (grid > 1))
    ):
        raise ValueError("grid must be a nonempty vector of points in [0, 1].")

    basis = cubic_basis(a.shape[1])
    design = basis(grid)
    weights = basis.integrate(0.0, 1.0)
    result = {"grid": grid, "summary_columns": np.array(["mean", "sd", "q025", "q975"])}
    mediated = a * b
    for name, paths in (("ME", mediated), ("DE", c), ("TE", mediated + c)):
        result["I" + name] = _summary(paths @ weights)
        curve = np.empty((len(grid), 4))
        # Keep memory bounded when many posterior draws are retained.
        for start in range(0, len(grid), 32):
            block = slice(start, start + 32)
            curve[block] = _summary(paths @ design[block].T)
        result[name] = curve
    return result


def effect_coefficient_mean(posterior, burnin):
    if burnin is None:
        raise ValueError("Explicit burn-in is required.")
    A = np.asarray(posterior["A"], float)
    B = np.asarray(posterior["B"], float)
    C = np.asarray(posterior["C"], float)
    nit = int(posterior.get("n_itrs", len(A) - 1))
    start = int(burnin) + 1 if len(A) == nit + 1 else int(burnin) if len(A) == nit else None
    if start is None or not 0 <= int(burnin) < nit or len(B) != len(A) or len(C) != len(A):
        raise ValueError("Cannot align population draws and burn-in.")
    if B.ndim != 3 or B.shape[1] != B.shape[2]:
        raise ValueError("B must contain matrix draws.")
    K = B.shape[1]
    A = A.reshape(len(A), K)[start:]
    C = C.reshape(len(C), K)[start:]
    B = B[start:]
    if not all(np.isfinite(x).all() for x in (A, B, C)):
        raise ValueError("Nonfinite posterior.")
    me = np.einsum("sij,sj->si", B, A)
    return {"ME": me.mean(0), "DE": C.mean(0), "TE": (me + C).mean(0)}
