"""Generate the finite-sample and aggregation datasets with paired donor subsets."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from blafm.mcmc import cubic_basis


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for the nested finite-sample recovery experiment."""

    sample_sizes: tuple[int, ...] = (200, 500, 1000)
    n_replicates: int = 10
    base_seed: int = 20260720

    # Fixed across all sample-size settings.
    min_cells: int = 10
    max_cells: int = 200

    # Coefficient-level data-generating model.
    basis_dimension: int = 5
    spline_degree: int = 3
    observation_variance_x: float = 0.01
    observation_variance_y: float = 0.01

    # True pseudotime is fixed to the deterministic equally spaced grid
    # used in the original data-generating code.
    pseudotime_design: str = "grid"

    # Dense grid for truth-based trajectory/effect evaluation.
    n_evaluation_grid: int = 501

    # Saving all donor trajectories on the dense grid is convenient for IMSE,
    # but increases disk use. It is enabled by default for this experiment.
    save_dense_trajectories: bool = True


# Intercepts and path coefficients in the two coefficient-basis designs.
def path_truth(k):
    if k == 4:
        alpha0 = [0.3, 0.4, 0.1, 0.1]
        beta0 = [0.2, 0.3, 0.2, 0.2]
        a, b, c = [0.4, 0.4, -0.4, -0.4], [1, 1, 0.25, 0.25], [0.2] * 4
    elif k == 5:
        alpha0 = [0.3, 0.4, 0.1, 0.1, 0.2]
        beta0 = [0.2, 0.3, 0.2, 0.2, 0.1]
        a, b, c = [0.3, 0.5, 0.8, 0.7, 0.8], [0.4, 0.5, 0.7, 0.6, 0.6], [0.3, 0.4, 0.2, 0.5, 0.7]
    else:
        raise ValueError("The finite-sample and aggregation designs use K=5 and K=4 respectively.")
    return np.array(alpha0), np.array(beta0), np.array(a), np.diag(b), np.array(c)


def validate_config(config: SimulationConfig) -> None:
    sizes = tuple(sorted(set(config.sample_sizes)))
    if sizes != config.sample_sizes:
        raise ValueError("sample_sizes must be unique and supplied in increasing order.")
    if any(n <= 0 for n in sizes):
        raise ValueError("All sample sizes must be positive.")
    if config.n_replicates <= 0:
        raise ValueError("n_replicates must be positive.")
    if config.min_cells < 2:
        raise ValueError("min_cells must be at least 2.")
    if config.max_cells < config.min_cells:
        raise ValueError("max_cells must be at least min_cells.")
    if config.spline_degree != 3:
        raise ValueError("This script is written for cubic B-splines (degree 3).")
    if config.pseudotime_design != "grid":
        raise ValueError("This script fixes pseudotime_design to 'grid'.")
    if config.observation_variance_x <= 0 or config.observation_variance_y <= 0:
        raise ValueError("Observation variances must be positive.")
    if config.n_evaluation_grid < 2:
        raise ValueError("n_evaluation_grid must be at least 2.")


def bspline_basis_matrix(
    t_grid: FloatArray,
    basis_dimension: int,
    degree: int = 3,
) -> FloatArray:
    """Evaluate the cubic B-spline basis used by the fitted model."""
    t_grid = np.asarray(t_grid, dtype=float)
    if np.any((t_grid < 0.0) | (t_grid > 1.0)):
        raise ValueError("All pseudotime values must lie in [0, 1].")

    if degree != 3:
        raise ValueError("The simulation uses cubic splines.")
    return cubic_basis(basis_dimension)(t_grid)


def generate_pseudotime(
    n_cells: int,
) -> tuple[FloatArray, IntArray]:
    """Generate equally spaced true pseudotime and coarse state labels."""
    n_early = n_cells // 2
    n_late = n_cells - n_early

    # Fixed design: equally spaced pseudotime over [0, 1].
    t = np.linspace(0.0, 1.0, n_cells, dtype=float)
    state = np.concatenate(
        [
            np.zeros(n_early, dtype=np.int64),
            np.ones(n_late, dtype=np.int64),
        ]
    )
    return t, state


def numerical_integral(values: FloatArray, grid: FloatArray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, grid))
    return float(np.trapz(values, grid))


def generate_master_dataset(
    config: SimulationConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Generate one n_max master dataset before donor subsampling."""
    n_max = max(config.sample_sizes)
    k = config.basis_dimension
    alpha0, beta0, a, b_matrix, c = path_truth(k)
    sigma, lam = np.eye(k) * 0.01, np.eye(k) * 0.0001

    z = rng.normal(0.0, 1.0, size=n_max)
    cell_counts = rng.integers(
        config.min_cells,
        config.max_cells + 1,
        size=n_max,
        dtype=np.int64,
    )

    eta = rng.multivariate_normal(np.zeros(k), sigma, size=n_max)
    xi = rng.multivariate_normal(np.zeros(k), lam, size=n_max)

    alpha = alpha0[None, :] + z[:, None] * a[None, :] + eta
    beta = beta0[None, :] + z[:, None] * c[None, :] + alpha @ b_matrix.T + xi

    x_all: list[list[float]] = []
    y_all: list[list[float]] = []
    t_all: list[list[float]] = []
    state_all: list[list[int]] = []
    g_all: list[list[float]] = []
    h_all: list[list[float]] = []

    evaluation_grid = np.linspace(0.0, 1.0, config.n_evaluation_grid)
    phi_grid = bspline_basis_matrix(
        evaluation_grid,
        basis_dimension=k,
        degree=config.spline_degree,
    )
    g_grid = alpha @ phi_grid.T
    h_grid = beta @ phi_grid.T

    sd_x = float(np.sqrt(config.observation_variance_x))
    sd_y = float(np.sqrt(config.observation_variance_y))

    for donor_index in range(n_max):
        n_cells = int(cell_counts[donor_index])
        t_i, state_i = generate_pseudotime(
            n_cells=n_cells,
        )
        phi_i = bspline_basis_matrix(
            t_i,
            basis_dimension=k,
            degree=config.spline_degree,
        )

        g_i = phi_i @ alpha[donor_index]
        h_i = phi_i @ beta[donor_index]
        x_i = g_i + rng.normal(0.0, sd_x, size=n_cells)
        y_i = h_i + rng.normal(0.0, sd_y, size=n_cells)

        x_all.append(x_i.tolist())
        y_all.append(y_i.tolist())
        t_all.append(t_i.tolist())
        state_all.append(state_i.tolist())
        g_all.append(g_i.tolist())
        h_all.append(h_i.tolist())

    mediated_coefficients = b_matrix @ a
    total_coefficients = mediated_coefficients + c
    me_true = phi_grid @ mediated_coefficients
    de_true = phi_grid @ c
    te_true = phi_grid @ total_coefficients

    effect_truth = {
        "grid": evaluation_grid.tolist(),
        "ME": me_true.tolist(),
        "DE": de_true.tolist(),
        "TE": te_true.tolist(),
        "IME": numerical_integral(me_true, evaluation_grid),
        "IDE": numerical_integral(de_true, evaluation_grid),
        "ITE": numerical_integral(te_true, evaluation_grid),
        "mediated_coefficient_vector": mediated_coefficients.tolist(),
        "direct_coefficient_vector": c.tolist(),
        "total_coefficient_vector": total_coefficients.tolist(),
    }

    coefficients = {
        "alpha0": alpha0.tolist(),
        "beta0": beta0.tolist(),
        "A": a.tolist(),
        "B": b_matrix.tolist(),
        "C": c.tolist(),
        "Sigma": sigma.tolist(),
        "Lambda": lam.tolist(),
        "basis_dimension": k,
        "spline_degree": config.spline_degree,
    }

    return {
        "Z": z,
        "C_cells": cell_counts,
        "X": x_all,
        "Y": y_all,
        "T": t_all,
        "cell_type": state_all,
        "g": g_all,
        "h": h_all,
        "alpha": alpha,
        "beta": beta,
        "grid": evaluation_grid,
        "g_grid": g_grid,
        "h_grid": h_grid,
        "effect_truth": effect_truth,
        "path_truth": coefficients,
    }


def subset_list(values: list[Any], donor_ids: IntArray) -> list[Any]:
    return [values[int(i)] for i in donor_ids]


def create_nested_subsets(
    master: dict[str, Any],
    sample_sizes: tuple[int, ...],
    rng: np.random.Generator,
) -> tuple[dict[int, dict[str, Any]], IntArray]:
    """Create nested donor subsets using one random donor permutation."""
    n_max = max(sample_sizes)
    donor_order = rng.permutation(n_max).astype(np.int64)
    subsets: dict[int, dict[str, Any]] = {}

    for n in sample_sizes:
        # Sorting makes saved donor order deterministic and easy to inspect.
        # Nestedness is preserved because all sets come from donor_order[:n].
        donor_ids = np.sort(donor_order[:n])
        subsets[n] = {
            "donor_ids": donor_ids,
            "Z": master["Z"][donor_ids],
            "C_cells": master["C_cells"][donor_ids],
            "X": subset_list(master["X"], donor_ids),
            "Y": subset_list(master["Y"], donor_ids),
            "T": subset_list(master["T"], donor_ids),
            "cell_type": subset_list(master["cell_type"], donor_ids),
            "g": subset_list(master["g"], donor_ids),
            "h": subset_list(master["h"], donor_ids),
            "alpha": master["alpha"][donor_ids],
            "beta": master["beta"][donor_ids],
            "g_grid": master["g_grid"][donor_ids],
            "h_grid": master["h_grid"][donor_ids],
        }

    # Explicit safety checks for the nested design.
    for smaller, larger in zip(sample_sizes[:-1], sample_sizes[1:]):
        small_ids = set(subsets[smaller]["donor_ids"].tolist())
        large_ids = set(subsets[larger]["donor_ids"].tolist())
        if not small_ids.issubset(large_ids):
            raise RuntimeError(f"n={smaller} is not nested within n={larger}.")

    return subsets, donor_order


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(value), file, ensure_ascii=False, indent=2)


def save_subset(
    subset: dict[str, Any],
    output_dir: Path,
    master: dict[str, Any],
    config: SimulationConfig,
    replicate_index: int,
    data_seed: int,
    subset_seed: int,
) -> None:
    generated_dir = output_dir / "generated_data"
    generated_dir.mkdir(parents=True, exist_ok=True)

    write_json(generated_dir / "X.json", subset["X"])
    write_json(generated_dir / "Y.json", subset["Y"])
    write_json(generated_dir / "Z.json", subset["Z"])
    write_json(generated_dir / "C.json", subset["C_cells"])
    write_json(generated_dir / "T.json", subset["T"])
    write_json(generated_dir / "cell_type.json", subset["cell_type"])

    # Noise-free truths at the observed cell pseudotimes.
    write_json(generated_dir / "g.json", subset["g"])
    write_json(generated_dir / "h.json", subset["h"])

    write_json(generated_dir / "alpha.json", subset["alpha"])
    write_json(generated_dir / "beta.json", subset["beta"])
    write_json(generated_dir / "path_truth.json", master["path_truth"])

    # Function-level truths used for RISE and integrated-effect recovery.
    write_json(generated_dir / "grid.json", master["grid"])
    write_json(generated_dir / "effect_truth.json", master["effect_truth"])

    if config.save_dense_trajectories:
        write_json(generated_dir / "g_grid.json", subset["g_grid"])
        write_json(generated_dir / "h_grid.json", subset["h_grid"])

    write_json(generated_dir / "donor_ids.json", subset["donor_ids"])

    metadata = {
        "replicate": replicate_index,
        "sample_size": int(len(subset["donor_ids"])),
        "master_sample_size": max(config.sample_sizes),
        "nested_sample_sizes": list(config.sample_sizes),
        "data_seed": data_seed,
        "subset_seed": subset_seed,
        "cell_count_distribution": {
            "name": "DiscreteUniform",
            "minimum": config.min_cells,
            "maximum": config.max_cells,
        },
        "exposure_distribution": "N(0,1)",
        "pseudotime_design": config.pseudotime_design,
        "observation_variance_x": config.observation_variance_x,
        "observation_variance_y": config.observation_variance_y,
        "basis_dimension": config.basis_dimension,
        "spline_degree": config.spline_degree,
    }
    write_json(generated_dir / "metadata.json", metadata)
    with (generated_dir / "seed.txt").open("w", encoding="utf-8") as file:
        file.write(f"data_seed={data_seed}\nsubset_seed={subset_seed}\n")


def generate_replicate(folder, config, replicate, sequence):
    """Build one paired nested replicate and publish it after all files are written."""
    from tempfile import TemporaryDirectory

    validate_config(config)
    data_sequence, subset_sequence = sequence.spawn(2)
    data_seed = int(data_sequence.generate_state(1)[0])
    subset_seed = int(subset_sequence.generate_state(1)[0])
    record = dict(
        design=asdict(config), replicate=replicate, data_seed=data_seed, subset_seed=subset_seed
    )
    record["design"]["sample_sizes"] = list(config.sample_sizes)
    if folder.exists():
        marker = folder / "replicate_metadata.json"
        if marker.is_file() and json.loads(marker.read_text()) == record:
            print(f"Using generated data in {folder}")
            return
        raise FileExistsError(f"Incomplete or different simulation at {folder}.")
    folder.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=folder.parent) as temporary:
        stage = Path(temporary) / folder.name
        stage.mkdir()
        master = generate_master_dataset(config, np.random.default_rng(data_sequence))
        subsets, order = create_nested_subsets(
            master, config.sample_sizes, np.random.default_rng(subset_sequence)
        )
        write_json(stage / "master_donor_order.json", order)
        for n, subset in subsets.items():
            save_subset(subset, stage / f"n_{n}", master, config, replicate, data_seed, subset_seed)
        write_json(stage / "replicate_metadata.json", record)
        stage.rename(folder)
    print(f"Generated {folder}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["finite", "aggregation"])
    parser.add_argument("--replicates", type=int, nargs="+")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    settings = json.loads((Path(__file__).resolve().parents[1] / "settings.json").read_text())
    design = settings[args.experiment]
    ids = settings["replicates"]
    reps = args.replicates or ids
    if len(set(reps)) != len(reps) or not set(reps) <= set(ids):
        parser.error("--replicates must be distinct IDs in the study configuration.")
    sizes = design["sample_sizes"] if args.experiment == "finite" else [design["n"]]
    config = SimulationConfig(
        sample_sizes=tuple(sizes),
        basis_dimension=design["K"],
        n_replicates=len(ids),
        base_seed=design["data_base_seed"],
    )
    output = args.output_root or Path("outputs") / args.experiment
    streams = np.random.SeedSequence(config.base_seed).spawn(len(ids))
    for rep in reps:
        generate_replicate(output / f"replicate_{rep:03d}", config, rep, streams[ids.index(rep)])


if __name__ == "__main__":
    main()
