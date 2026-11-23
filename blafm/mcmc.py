"""Conditional distributions for the diagonal-B BLaFM sampler."""

import numpy as np
from scipy.stats import invgamma, geninvgauss
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.interpolate import BSpline
from scipy.special import ndtr, ndtri
from functools import lru_cache


# Basic helpers


def _valid_mask_from_Cd(C_d, c_max):
    return np.arange(c_max)[None, :] < np.asarray(C_d)[:, None]


def _pad_cell_type(cell_type, C_d, c_max, n):
    ct = np.full((n, c_max), -1, dtype=np.int8)
    for i, ci in enumerate(C_d):
        ct[i, :ci] = np.asarray(cell_type[i], dtype=np.int8)[:ci]
    return ct


# Exact GIG sampler
# density proportional to x^{lam-1} exp(-(psi*x + chi/x)/2)


def sample_gig(lam, psi, chi, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    lam = float(lam)
    psi = float(psi)
    chi = float(chi)
    if psi <= 0 or chi <= 0:
        raise ValueError(f"GIG requires psi>0 and chi>0, got psi={psi}, chi={chi}")

    b = np.sqrt(psi * chi)
    scale = np.sqrt(chi / psi)
    return geninvgauss.rvs(lam, b, scale=scale, random_state=rng)


# Observation-level variance updates


def update_sigma_i(X, n, C_d, alpha, phi, a1, b1, random_state=None):
    X = np.asarray(X, dtype=float)
    phi = np.asarray(phi, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    c_max = X.shape[1]
    valid_mask = _valid_mask_from_Cd(C_d, c_max)

    g = np.einsum("ijk,ik->ij", phi, alpha)
    resid2 = (X - g) ** 2
    sse = np.sum(np.where(valid_mask, resid2, 0.0), axis=1)

    a_vec = a1 + np.asarray(C_d, dtype=float) / 2.0
    b_vec = b1 + sse / 2.0
    if np.any(b_vec <= 0):
        bad = np.where(b_vec <= 0)[0]
        raise ValueError(f"Non-positive scale in update_sigma_i at indices {bad.tolist()}")

    return np.atleast_1d(invgamma.rvs(a=a_vec, scale=b_vec, random_state=random_state))


def update_tau_i(Y, n, C_d, beta, phi, a2, b2, random_state=None):
    Y = np.asarray(Y, dtype=float)
    phi = np.asarray(phi, dtype=float)
    beta = np.asarray(beta, dtype=float)
    c_max = Y.shape[1]
    valid_mask = _valid_mask_from_Cd(C_d, c_max)

    h = np.einsum("ijk,ik->ij", phi, beta)
    resid2 = (Y - h) ** 2
    sse = np.sum(np.where(valid_mask, resid2, 0.0), axis=1)

    a_vec = a2 + np.asarray(C_d, dtype=float) / 2.0
    b_vec = b2 + sse / 2.0
    if np.any(b_vec <= 0):
        bad = np.where(b_vec <= 0)[0]
        raise ValueError(f"Non-positive scale in update_tau_i at indices {bad.tolist()}")

    return np.atleast_1d(invgamma.rvs(a=a_vec, scale=b_vec, random_state=random_state))


# Joint update of (alpha_i, beta_i) with intercepts
# alpha_i ~ N(alpha0 + A*z_i, Sigma)
# beta_i  ~ N(beta0 + C*z_i + B alpha_i, Lambda)


def update_alpha_beta_block(
    alpha0,
    A,
    B,
    beta0,
    C,
    X_,
    Y_,
    Z,
    phi,
    phi_phi_t,
    Sigma,
    Lambda,
    sigma_i,
    tau_i,
    n,
    random_state=None,
):
    random = np.random if random_state is None else random_state
    alpha0 = np.asarray(alpha0, dtype=float).reshape(-1)
    A = np.asarray(A, dtype=float).reshape(-1)
    B = np.asarray(B, dtype=float)
    beta0 = np.asarray(beta0, dtype=float).reshape(-1)
    C = np.asarray(C, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)
    Lambda = np.asarray(Lambda, dtype=float)
    Z = np.asarray(Z, dtype=float).reshape(-1)
    sigma_i = np.asarray(sigma_i, dtype=float).reshape(-1)
    tau_i = np.asarray(tau_i, dtype=float).reshape(-1)
    X_ = np.asarray(X_, dtype=float)
    Y_ = np.asarray(Y_, dtype=float)
    phi = np.asarray(phi, dtype=float)
    phi_phi_t = np.asarray(phi_phi_t, dtype=float)

    K = A.shape[0]
    new_alpha = np.empty((n, K), dtype=float)
    new_beta = np.empty((n, K), dtype=float)

    Sigma_inv = np.linalg.inv(Sigma)
    Lambda_inv = np.linalg.inv(Lambda)
    Bt_Linv_B = B.T @ Lambda_inv @ B

    xPhi_all = np.einsum("ij,ijk->ik", X_, phi)
    yPhi_all = np.einsum("ij,ijk->ik", Y_, phi)

    for i in range(n):
        Zi = float(Z[i])
        PhiTPhi = phi_phi_t[i]
        m_a = alpha0 + A * Zi
        m_b = beta0 + C * Zi

        Qaa = Sigma_inv + PhiTPhi / sigma_i[i] + Bt_Linv_B
        Qbb = Lambda_inv + PhiTPhi / tau_i[i]
        Qab = -B.T @ Lambda_inv
        Qba = -Lambda_inv @ B
        Q = np.block([[Qaa, Qab], [Qba, Qbb]])

        qa = xPhi_all[i] / sigma_i[i] + Sigma_inv @ m_a - B.T @ Lambda_inv @ m_b
        qb = yPhi_all[i] / tau_i[i] + Lambda_inv @ m_b
        q = np.concatenate([qa, qb], axis=0)

        cfac, lower = cho_factor(Q, lower=True, check_finite=False)
        mu = cho_solve((cfac, lower), q, check_finite=False)
        z = random.normal(size=2 * K)
        eps = solve_triangular(cfac.T, z, lower=False, check_finite=False)
        draw = mu + eps

        new_alpha[i] = draw[:K]
        new_beta[i] = draw[K:]

    return new_alpha, new_beta


# Exact conjugate updates for (alpha0_k, a_k)
# prior: d_k = (alpha0_k, a_k)^T ~ N(0, tau_a * Sigma_kk * I_2)


def update_alpha0_A(alpha, Z, Sigma, n, K, tau_a, random_state=None):
    random = np.random if random_state is None else random_state
    alpha = np.asarray(alpha, dtype=float)
    Z = np.asarray(Z, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)

    G = np.column_stack([np.ones_like(Z), Z])
    GTG = G.T @ G
    M = GTG + np.eye(2) / tau_a
    cfac, lower = cho_factor(M, lower=True, check_finite=False)

    new_alpha0 = np.empty(K, dtype=float)
    new_A = np.empty(K, dtype=float)

    for k in range(K):
        yk = alpha[:, k]
        mu_k = cho_solve((cfac, lower), G.T @ yk, check_finite=False)
        z = random.normal(size=2)
        eps = solve_triangular(cfac.T, z, lower=False, check_finite=False)
        draw = mu_k + np.sqrt(Sigma[k, k]) * eps
        new_alpha0[k] = draw[0]
        new_A[k] = draw[1]

    return new_alpha0.reshape(1, -1), new_A.reshape(1, -1)


# Exact conjugate updates for (beta0_k, c_k, b_k)
# prior: d_k = (beta0_k, c_k, b_k)^T ~ N(0, tau_d * Lambda_kk * I_3)


def update_beta0_BC_diag(Z, beta, alpha, Lambda, n, K, tau_d, random_state=None):
    random = np.random if random_state is None else random_state
    Z = np.asarray(Z, dtype=float).reshape(-1)
    beta = np.asarray(beta, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    Lambda = np.asarray(Lambda, dtype=float)

    new_d = np.empty((K, 3), dtype=float)

    for k in range(K):
        G = np.column_stack([np.ones_like(Z), Z, alpha[:, k]])
        M = G.T @ G + np.eye(3) / tau_d
        cfac, lower = cho_factor(M, lower=True, check_finite=False)
        mu_k = cho_solve((cfac, lower), G.T @ beta[:, k], check_finite=False)
        z = random.normal(size=3)
        eps = solve_triangular(cfac.T, z, lower=False, check_finite=False)
        draw = mu_k + np.sqrt(Lambda[k, k]) * eps
        new_d[k] = draw

    return new_d


# Exact full conditionals for Sigma and Lambda
# including upper-level Gaussian prior terms


def update_Sigma(alpha, alpha0, A, Z, n, K, tau_a, a, b, random_state=None):
    alpha = np.asarray(alpha, dtype=float)
    alpha0 = np.asarray(alpha0, dtype=float).reshape(-1)
    A = np.asarray(A, dtype=float).reshape(-1)
    Z = np.asarray(Z, dtype=float).reshape(-1)

    new_Sigma = np.empty(K, dtype=float)
    a_post = a + n / 2.0 + 1.0
    for k in range(K):
        resid = alpha[:, k] - alpha0[k] - A[k] * Z
        scale_post = b + 0.5 * np.sum(resid**2) + 0.5 * (alpha0[k] ** 2 + A[k] ** 2) / tau_a
        new_Sigma[k] = invgamma.rvs(a=a_post, scale=scale_post, random_state=random_state)
    return np.diag(new_Sigma)


def update_Lambda(alpha, beta, beta0, B, C, Z, n, K, tau_d, a, b, random_state=None):
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    beta0 = np.asarray(beta0, dtype=float).reshape(-1)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float).reshape(-1)
    Z = np.asarray(Z, dtype=float).reshape(-1)
    b_diag = np.diag(B)

    new_Lambda = np.empty(K, dtype=float)
    a_post = a + n / 2.0 + 1.5
    for k in range(K):
        resid = beta[:, k] - beta0[k] - C[k] * Z - b_diag[k] * alpha[:, k]
        scale_post = (
            b + 0.5 * np.sum(resid**2) + 0.5 * (beta0[k] ** 2 + C[k] ** 2 + b_diag[k] ** 2) / tau_d
        )
        new_Lambda[k] = invgamma.rvs(a=a_post, scale=scale_post, random_state=random_state)
    return np.diag(new_Lambda)


# Exact GIG full conditionals for tau_a and tau_d


def update_tau_a(a_a, b_a, alpha0, A, Sigma, K, rng=None):
    alpha0 = np.asarray(alpha0, dtype=float).reshape(-1)
    A = np.asarray(A, dtype=float).reshape(-1)
    sigma_diag = np.diag(np.asarray(Sigma, dtype=float))
    chi = np.sum((alpha0**2 + A**2) / sigma_diag)
    psi = 2.0 * b_a
    lam = float(a_a) - float(K)
    return sample_gig(lam=lam, psi=psi, chi=chi, rng=rng)


def update_tau_d(a_b, b_b, beta0, C, B, Lambda, K, rng=None):
    beta0 = np.asarray(beta0, dtype=float).reshape(-1)
    C = np.asarray(C, dtype=float).reshape(-1)
    b_diag = np.diag(np.asarray(B, dtype=float))
    lambda_diag = np.diag(np.asarray(Lambda, dtype=float))
    chi = np.sum((beta0**2 + C**2 + b_diag**2) / lambda_diag)
    psi = 2.0 * b_b
    lam = float(a_b) - 1.5 * float(K)
    return sample_gig(lam=lam, psi=psi, chi=chi, rng=rng)


# B-spline basis helpers


def cubic_basis(k):
    """Return all k clamped cubic basis functions as one SciPy spline."""
    return _vector_basis(tuple(np.linspace(0, 1, k - 2)), 3)


@lru_cache(maxsize=32)
def _vector_basis(breaks, degree):
    knots = np.r_[[breaks[0]] * (degree + 1), breaks[1:-1], [breaks[-1]] * (degree + 1)]
    return BSpline(knots, np.eye(len(knots) - degree - 1), degree, extrapolate=False)


def basis_design(grid, k, degree=3):
    """Evaluate k clamped basis functions on the study grid."""
    if k < degree + 1:
        raise ValueError("The number of basis functions must exceed the degree.")
    knots = np.linspace(0.0, 1.0, k - degree + 1)
    design = bspline_basis_matrix(grid, knots, degree)
    if not np.isfinite(design).all():
        raise ValueError("The evaluation grid must lie within the spline support [0, 1].")
    return design


def bspline_basis_matrix(t_grid, knots, degree=3):
    t_grid = np.asarray(t_grid, dtype=float)
    return _vector_basis(tuple(knots), degree)(t_grid)


def compute_bspline_basis_ragged(t_list, knot_vector, degree=3):
    t_list = [np.asarray(t, dtype=float) for t in t_list]
    lengths = [len(t) for t in t_list]
    t_flat = np.concatenate(t_list)

    phi_flat = bspline_basis_matrix(t_grid=t_flat, knots=knot_vector, degree=degree)

    phi_list = []
    idx = 0
    for n_i in lengths:
        phi_list.append(phi_flat[idx : idx + n_i])
        idx += n_i

    return phi_list


def pad_phi(phi_i, target_len):
    n_i, K = phi_i.shape
    if n_i > target_len:
        raise ValueError("Target length is smaller than the number of cells.")

    padded = np.zeros((target_len, K), dtype=phi_i.dtype)
    padded[:n_i, :] = phi_i
    return padded


# T updates


def _sample_truncnorm_loc(mu, low, high, sigma, rng):
    a = (low - mu) / sigma
    b = (high - mu) / sigma

    Phi_a = ndtr(a)
    Phi_b = ndtr(b)
    diff = Phi_b - Phi_a

    u = rng.random(mu.shape)
    z = Phi_a + u * diff

    eps = np.finfo(float).eps
    z = np.clip(z, eps, 1.0 - eps)

    x = mu + sigma * ndtri(z)
    return np.clip(x, low, high)


def _log_truncnorm_q(x, mu, low, high, sigma):
    a = (low - mu) / sigma
    b = (high - mu) / sigma
    Z = ndtr(b) - ndtr(a)
    z = (x - mu) / sigma
    return -0.5 * z * z - np.log(sigma) - np.log(Z + 1e-300)


def update_T_A_fast(
    X_,
    Y_,
    alpha,
    beta,
    phi,
    T_,
    sigma_i,
    tau_i,
    n,
    c_max,
    C_d,
    knots,
    K,
    cell_type,
    sigma=1.0,
    rng=None,
    use_q=True,
    cell_type_pad=None,
    return_phi=True,
):
    if rng is None:
        rng = np.random.default_rng()

    T_ = np.asarray(T_, dtype=float)
    phi = np.asarray(phi, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    X_ = np.asarray(X_, dtype=float)
    Y_ = np.asarray(Y_, dtype=float)
    sigma_col = np.asarray(sigma_i, dtype=float)[:, None]
    tau_col = np.asarray(tau_i, dtype=float)[:, None]

    valid_mask = _valid_mask_from_Cd(C_d, c_max)

    gt = np.einsum("ijk,ik->ij", phi, alpha)
    ht = np.einsum("ijk,ik->ij", phi, beta)

    if cell_type_pad is None:
        ct = _pad_cell_type(cell_type, C_d, c_max, n)
    else:
        ct = cell_type_pad

    low = np.where(ct == 0, 0.0, 0.5)
    high = np.where(ct == 0, 0.5, 1.0)
    low = np.where(valid_mask, low, 0.0)
    high = np.where(valid_mask, high, 1.0)

    new_T = _sample_truncnorm_loc(T_, low, high, sigma, rng)
    new_T = np.where(valid_mask, new_T, T_)

    T_list = [new_T[i, :c_i] for i, c_i in enumerate(C_d)]
    new_phi = compute_bspline_basis_ragged(T_list, knots, degree=3)
    new_phi = [pad_phi(arr, target_len=c_max) for arr in new_phi]
    new_phi = np.asarray(new_phi)

    gt_new = np.einsum("ijk,ik->ij", new_phi, alpha)
    ht_new = np.einsum("ijk,ik->ij", new_phi, beta)

    logp = -0.5 * (X_ - gt) ** 2 / sigma_col - 0.5 * (Y_ - ht) ** 2 / tau_col
    logp_new = -0.5 * (X_ - gt_new) ** 2 / sigma_col - 0.5 * (Y_ - ht_new) ** 2 / tau_col
    t = logp_new - logp

    if use_q:
        logq_fwd = _log_truncnorm_q(new_T, T_, low, high, sigma)
        logq_rev = _log_truncnorm_q(T_, new_T, low, high, sigma)
        t = t + (logq_rev - logq_fwd)

    u = np.log(rng.random(t.shape) + 1e-300)
    accept = (u < t) & valid_mask
    out_T = np.where(accept, new_T, T_)

    if return_phi:
        out_phi = np.where(accept[:, :, None], new_phi, phi)
        out_phi_phi_t = np.einsum("ikj,ijh->ikh", np.transpose(out_phi, axes=(0, 2, 1)), out_phi)
        return out_T, out_phi, out_phi_phi_t, 0, 0, 0, accept

    return out_T, 0, 0, 0, accept
