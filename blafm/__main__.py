"""Fit BLaFM to donor-grouped data and write posterior draws and effects."""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from . import fit, summarize_effects


def _read_json(path):
    with path.open(encoding="utf-8-sig") as file:
        return json.load(file)


def _load(path):
    if path.is_dir():
        fields = {"x": "X", "y": "Y", "z": "Z", "cell_type": "cell_type"}
        data = {key: _read_json(path / f"{stem}.json") for key, stem in fields.items()}
        for key in ("donor_ids", "cell_ids"):
            extra = path / f"{key}.json"
            if extra.is_file():
                data[key] = _read_json(extra)
    else:
        data = _read_json(path)
    if not isinstance(data, dict) or not {"x", "y", "z", "cell_type"}.issubset(data):
        raise ValueError("Input must provide x, y, z and cell_type.")
    return data


def _identifiers(data):
    """Retain supplied identifiers in the same donor/cell order as the estimates."""
    ids = {}
    donors = data.get("donor_ids", data.get("donor_id"))
    if donors is not None:
        donors = np.asarray(donors, dtype=str)
        if donors.shape != (len(data["x"]),) or len(np.unique(donors)) != len(donors):
            raise ValueError("donor_ids must give a distinct ID for each donor.")
        ids["donor_ids"] = donors
    if "cell_ids" in data:
        groups = data["cell_ids"]
        if len(groups) != len(data["x"]):
            raise ValueError("cell_ids must have one row per donor.")
        rows = []
        for labels, x in zip(groups, data["x"]):
            row = np.asarray(labels, dtype=str)
            if row.shape != (len(x),) or len(np.unique(row)) != len(x):
                raise ValueError("cell_ids must identify each cell uniquely within its donor.")
            rows.append(row)
        if rows:
            ids["cell_ids"] = np.concatenate(rows)
    return ids


def _save(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as temp:
        temporary_path = Path(temp.name)
    try:
        np.savez_compressed(temporary_path, **arrays)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Donor-major JSON or simulation data directory.")
    parser.add_argument(
        "--out", type=Path, default=Path("results/fit.npz"), help="Output NPZ file."
    )
    parser.add_argument("--K", type=int, default=5, help="Number of cubic spline basis functions.")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--burnin", type=int, help="Initial iterations to discard (default: half).")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--z-offset",
        type=float,
        default=0.0,
        help="Add to the supplied exposure. Use -1 for legacy 1/2/3 dosage.",
    )
    parser.add_argument("--quiet", action="store_true", help="Hide the MCMC progress bar.")
    args = parser.parse_args()
    if args.input.resolve() == args.out.resolve() or args.out.suffix.lower() != ".npz":
        parser.error("--out must name a .npz file distinct from the input.")

    try:
        data = _load(args.input)
        ids = _identifiers(data)
        z = np.asarray(data["z"], dtype=float) + args.z_offset
        draws = fit(
            data["x"],
            data["y"],
            z,
            data["cell_type"],
            K=args.K,
            iterations=args.iterations,
            burnin=args.burnin,
            seed=args.seed,
            progress=not args.quiet,
        )
        summaries = summarize_effects(draws)
        _save(args.out, {**draws, **summaries, **ids, "z_offset": args.z_offset})
    except (ValueError, OSError, TypeError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")

    print("Effect      Mean        SD      2.5%     97.5%")
    for name in ("IME", "IDE", "ITE"):
        values = " ".join(f"{v:9.4f}" for v in summaries[name])
        print(f"{name}    {values}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
