"""Shared cell folds, conditional log predictive densities and result I/O."""

from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import numpy as np
from blafm.mcmc import bspline_basis_matrix

LOG_2PI = math.log(2 * math.pi)


def split_one_fold(
    X: list[Any],
    Y: list[Any],
    cell_type: list[Any],
    fold_ids: list[np.ndarray],
    heldout_fold_zero_based: int,
) -> tuple[list[list[float]], list[list[float]], list[list[int]], dict[str, np.ndarray]]:
    X_train: list[list[float]] = []
    Y_train: list[list[float]] = []
    cell_type_train: list[list[int]] = []

    heldout_donor: list[int] = []
    heldout_cell: list[int] = []
    heldout_state: list[int] = []
    heldout_x: list[float] = []
    heldout_y: list[float] = []

    for donor_index, (x_i, y_i, ct_i, fold_i) in enumerate(zip(X, Y, cell_type, fold_ids)):
        x = np.asarray(x_i, dtype=float)
        y = np.asarray(y_i, dtype=float)
        ct = np.asarray(ct_i, dtype=np.int8)
        fold_i = np.asarray(fold_i, dtype=np.int16)

        test_mask = fold_i == heldout_fold_zero_based
        train_mask = ~test_mask

        if np.sum(train_mask) == 0:
            raise ValueError(
                f"Donor {donor_index} has no training cells in fold {heldout_fold_zero_based + 1}."
            )

        X_train.append(x[train_mask].tolist())
        Y_train.append(y[train_mask].tolist())
        cell_type_train.append(ct[train_mask].astype(int).tolist())

        for original_cell_index in np.flatnonzero(test_mask):
            heldout_donor.append(donor_index)
            heldout_cell.append(int(original_cell_index))
            heldout_state.append(int(ct[original_cell_index]))
            heldout_x.append(float(x[original_cell_index]))
            heldout_y.append(float(y[original_cell_index]))

    if not heldout_x:
        raise ValueError("The selected fold contains no held-out cells.")

    heldout = {
        "donor_index": np.asarray(heldout_donor, dtype=np.int64),
        "cell_index": np.asarray(heldout_cell, dtype=np.int64),
        "state": np.asarray(heldout_state, dtype=np.int8),
        "x": np.asarray(heldout_x, dtype=float),
        "y": np.asarray(heldout_y, dtype=float),
    }
    return X_train, Y_train, cell_type_train, heldout


def normal_logpdf_from_variance(observed, mean, variance):
    variance = np.asarray(variance, dtype=float)
    if np.any(~np.isfinite(variance)) or np.any(variance <= 0.0):
        raise FloatingPointError("Encountered a nonpositive or nonfinite variance.")
    return -0.5 * (LOG_2PI + np.log(variance) + (observed - mean) ** 2 / variance)


def build_fixed_test_basis(heldout, fixed_times, knots, degree=3):
    """Align the ORIGINAL cell-specific T_pred with held-out rows and evaluate once."""
    t_hat = np.asarray(
        [
            fixed_times[int(i)][int(c)]
            for i, c in zip(heldout["donor_index"], heldout["cell_index"])
        ],
        dtype=float,
    )
    phi_test = bspline_basis_matrix(t_hat, knots, degree=degree)
    if len(t_hat) != len(heldout["x"]) or not np.isfinite(phi_test).all():
        raise ValueError("Invalid fixed test times or B-spline design.")
    t_hat.flags.writeable = False
    phi_test.flags.writeable = False
    heldout["t_hat"] = t_hat
    return phi_test


def update_conditional_log_accumulators(
    heldout,
    phi_test,
    alpha,
    beta,
    sigma_i,
    tau_i,
    log_accumulator_x,
    log_accumulator_xy,
):
    """One posterior draw at each fixed test t; NO integration over time."""
    donor = heldout["donor_index"]
    mu_x = np.einsum("ij,ij->i", phi_test, alpha[donor])
    mu_y = np.einsum("ij,ij->i", phi_test, beta[donor])
    log_fx = normal_logpdf_from_variance(heldout["x"], mu_x, sigma_i[donor])
    log_fy = normal_logpdf_from_variance(heldout["y"], mu_y, tau_i[donor])
    # Sum densities online in log space; divide by the number of draws at the end.
    np.logaddexp(log_accumulator_x, log_fx, out=log_accumulator_x)
    np.logaddexp(log_accumulator_xy, log_fx + log_fy, out=log_accumulator_xy)


def write_trace(path: Path, trace_rows: list[dict[str, Any]]) -> None:
    if not trace_rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(trace_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trace_rows)


def summarize_fold_scores(
    heldout: dict[str, np.ndarray],
    lpd_x: np.ndarray,
    lpd_xy: np.ndarray,
) -> dict[str, Any]:
    lpd_y_given_x = lpd_xy - lpd_x
    donor_indices = np.unique(heldout["donor_index"])

    donor_mean_x = []
    donor_mean_y_given_x = []
    donor_mean_xy = []
    for donor_index in donor_indices:
        mask = heldout["donor_index"] == donor_index
        donor_mean_x.append(float(np.mean(lpd_x[mask])))
        donor_mean_y_given_x.append(float(np.mean(lpd_y_given_x[mask])))
        donor_mean_xy.append(float(np.mean(lpd_xy[mask])))

    return {
        "n_heldout_cells": int(len(lpd_xy)),
        "n_heldout_donors": int(len(donor_indices)),
        "total_lpd_x": float(np.sum(lpd_x)),
        "total_lpd_y_given_x": float(np.sum(lpd_y_given_x)),
        "total_lpd_xy": float(np.sum(lpd_xy)),
        "mean_cell_lpd_x": float(np.mean(lpd_x)),
        "mean_cell_lpd_y_given_x": float(np.mean(lpd_y_given_x)),
        "mean_cell_lpd_xy": float(np.mean(lpd_xy)),
        "donor_balanced_mean_cell_lpd_x": float(np.mean(donor_mean_x)),
        "donor_balanced_mean_cell_lpd_y_given_x": float(np.mean(donor_mean_y_given_x)),
        "donor_balanced_mean_cell_lpd_xy": float(np.mean(donor_mean_xy)),
        "max_abs_decomposition_error": float(np.max(np.abs(lpd_xy - (lpd_x + lpd_y_given_x)))),
    }


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(value), ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def save_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(
        json.dumps(jsonable(value), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def vector_hash(rows, dtype="<f8"):
    h = hashlib.sha256()
    for row in rows:
        a = np.asarray(row, dtype=dtype)
        h.update(np.asarray([len(a)], dtype="<i8").tobytes())
        h.update(a.tobytes())
    return h.hexdigest()


def integer(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: expected an integer, got {value!r}") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{label}: expected a finite integer, got {value!r}")
    return int(number)
