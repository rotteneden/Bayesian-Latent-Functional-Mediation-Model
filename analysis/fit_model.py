"""Full joint or known-state BLaFM fit with manuscript priors and explicit RNGs.

The numerical update functions come from the supplied sampler. Population
chains include the initial state at index 0; auxiliary draws have explicit
one-based iteration numbers. Only actual post-burn-in draws enter means.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import scipy
from blafm import mcmc
from blafm.model import sample_chain
from blafm import model as core_model

ROOT = Path(__file__).resolve().parents[1]
POPULATION = ("alpha0", "A", "beta0", "B", "C")
AUXILIARY = ("Sigma", "Lambda", "sigma_i", "tau_i", "tau_a", "tau_d")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_value(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(f"Cannot encode {type(x).__name__}")


def atomic_json(path, value):
    from tempfile import NamedTemporaryFile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as f:
        temp = Path(f.name)
        try:
            json.dump(value, f, ensure_ascii=False, allow_nan=False, default=json_value)
            f.close()
            temp.replace(path)
        finally:
            f.close()
            temp.unlink(missing_ok=True)


def load_data(args):
    ids = {}
    if args.application_data:
        path = Path(args.application_data).resolve()
        doc = read_json(path)
        X, Y, Z, state = doc["x"], doc["y"], doc["z"], doc["cell_type"]
        Z = np.asarray(Z, dtype=float).reshape(-1) + args.z_offset
        if not np.isin(Z, [0, 1, 2]).all():
            raise ValueError("Application z after z_offset must be G-allele dosage 0/1/2.")
        hashes = digest(path)
        if "donor_ids" not in doc or "cell_ids" not in doc:
            raise ValueError(
                "Application data need donor_ids and cell_ids. Use python -m analysis.application.prepare."
            )
        ids = {k: doc[k] for k in ("donor_ids", "cell_ids")}
    else:
        path = Path(args.data).resolve()
        X, Y, Z, state = [read_json(path / (k + ".json")) for k in ("X", "Y", "Z", "cell_type")]
        hashes = {k: digest(path / (k + ".json")) for k in ("X", "Y", "Z", "cell_type")}
        Z = np.asarray(Z, dtype=float).reshape(-1)
    n = len(X)
    if not n or len(Y) != n or len(state) != n or len(Z) != n or not np.isfinite(Z).all():
        raise ValueError("Empty data or inconsistent donor counts/nonfinite exposure.")
    counts = np.asarray([len(r) for r in X], dtype=int)
    if np.any(counts < 1):
        raise ValueError("Empty donor.")
    for i in range(n):
        for label, r in [("X", X[i]), ("Y", Y[i]), ("state", state[i])]:
            a = np.asarray(r, dtype=float)
            if a.shape != (counts[i],) or not np.isfinite(a).all():
                raise ValueError(f"{label}[{i}]: invalid row or nonfinite values.")
        if not np.isin(state[i], [0, 1]).all():
            raise ValueError("States must be 0/1.")
    if ids:
        donors = [str(x) for x in ids["donor_ids"]]
        if len(donors) != n or len(set(donors)) != n:
            raise ValueError("Invalid donor IDs.")
        if len(ids["cell_ids"]) != n:
            raise ValueError("Invalid cell IDs.")
        for row, nc in zip(ids["cell_ids"], counts):
            if len(row) != nc or len(set(map(str, row))) != nc:
                raise ValueError("Invalid cell IDs.")
    return X, Y, Z, state, counts, path, hashes, ids


def fit(args):
    if not 0 <= args.burnin < args.iterations - 1 or args.aux_thin < 1:
        raise ValueError("Require >=2 post-burn-in draws and aux_thin>=1.")
    if args.K < 4 or not 0 <= args.seed < 2**32:
        raise ValueError("Invalid K or seed.")
    X, Y, Z, state, counts, data_path, hashes, ids = load_data(args)
    out = Path(args.output).resolve()
    settings = read_json(args.paper_config)
    hp = settings["hyperparameters"]
    needed = {f"{p}{i}" for p in "ab" for i in range(1, 5)} | {"a_a", "b_a", "a_b", "b_b"}
    if set(hp) != needed or any(not np.isfinite(v) or v <= 0 for v in hp.values()):
        raise ValueError("Invalid hyperparameters.")
    n, K = len(counts), args.K
    if args.oracle and args.application_data:
        raise ValueError("Known states are available only for simulation.")
    fixed_times = read_json(data_path / "T.json") if args.oracle else None
    posterior_path, config_path = out / "posterior_set.json", out / "fit_config.json"
    request = dict(
        K=K,
        n=n,
        replicate=args.replicate,
        iterations=args.iterations,
        burnin=args.burnin,
        seed=args.seed,
        aux_thin=args.aux_thin,
        data_sha256=hashes,
        hyperparameters=hp,
        oracle_pseudotime=args.oracle,
        known_states_sha256=digest(data_path / "T.json") if args.oracle else None,
        dosage_offset=args.z_offset if args.application_data else None,
        sampler_sha256=digest(mcmc.__file__),
        driver_sha256=digest(__file__),
        chain_sha256=digest(core_model.__file__),
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
    )
    if posterior_path.exists() or config_path.exists():
        if getattr(args, "resume", False) and config_path.is_file() and posterior_path.is_file():
            cached = read_json(config_path)
            if all(cached.get(k) == v for k, v in request.items()) and cached.get(
                "posterior_sha256"
            ) == digest(posterior_path):
                print(f"Using completed fit in {out}")
                return None, cached
        if not args.overwrite:
            raise FileExistsError(
                f"Prior or incomplete fit at {out}. Use a new destination for a changed run."
            )
    started = time.perf_counter()
    chain = sample_chain(
        X,
        Y,
        Z,
        state,
        K=K,
        iterations=args.iterations,
        seed=args.seed,
        progress=not args.quiet,
        fixed_times=fixed_times,
        priors=hp,
    )
    initial = next(chain)
    kept = args.iterations - args.burnin
    p = {key: [value] for key, value in zip(POPULATION, initial.population)}
    p.update({key: [] for key in AUXILIARY})
    p.update(
        K=K,
        degree=3,
        burnin=args.burnin,
        n_itrs=args.iterations,
        aux_thin=args.aux_thin,
        aux_saved_itr=[],
        T_pred=np.zeros_like(initial.times),
        alpha_pred=np.zeros_like(initial.alpha),
        beta_pred=np.zeros_like(initial.beta),
        pos_inds=np.zeros_like(initial.times),
        **ids,
    )
    p["init"] = dict(K=K, degree=3, burnin=args.burnin, n_itrs=args.iterations, **hp)
    if args.application_data:
        p["z_offset"] = args.z_offset
    if args.oracle:
        p["conditioning"] = "true_states"
    for itr, draw in enumerate(chain, start=1):
        for key, value in zip(POPULATION, draw.population):
            p[key].append(value)
        if itr <= args.burnin:
            continue
        p["T_pred"] += draw.times
        p["alpha_pred"] += draw.alpha
        p["beta_pred"] += draw.beta
        p["pos_inds"] += draw.accepted
        if (itr - args.burnin - 1) % args.aux_thin == 0:
            p["aux_saved_itr"].append(itr)
            for key, value in zip(AUXILIARY, draw.auxiliary):
                p[key].append(value)
    for key in ("T_pred", "alpha_pred", "beta_pred"):
        p[key] /= kept
    for key in POPULATION:
        p[key + "_HAT"] = np.mean(p[key][args.burnin + 1 :], axis=0)
    p["T"] = [[draw.times[i, :nc] for i, nc in enumerate(counts)]]
    p["alpha"], p["beta"] = [draw.alpha], [draw.beta]
    p["data_sha256"] = hashes
    cfg = dict(
        request,
        degree=3,
        effective_seed=args.seed,
        data=str(data_path),
        elapsed_seconds=time.perf_counter() - started,
        population_initial_state_saved=True,
    )
    if args.application_data:
        cfg["z_offset"] = args.z_offset
    atomic_json(out / "posterior_set.json", p)
    cfg["posterior_sha256"] = digest(out / "posterior_set.json")
    atomic_json(out / "fit_config.json", cfg)
    print(f"Saved {out / 'posterior_set.json'}")
    return p, cfg


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--data", type=Path)
    g.add_argument("--application-data", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--replicate", type=int, default=None)
    p.add_argument("--iterations", type=int, default=10000)
    p.add_argument("--burnin", type=int, default=5000)
    p.add_argument("--aux-thin", type=int, default=20)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--z-offset", type=float, default=0.0)
    p.add_argument("--paper-config", type=Path, default=ROOT / "analysis/settings.json")
    p.add_argument("--oracle", action="store_true")
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--resume", action="store_true")
    modes.add_argument("--overwrite", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


if __name__ == "__main__":
    fit(parser().parse_args())
