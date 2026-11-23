from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for one independently generated dataset."""

    n_donors: int = 500

    # C_i ~ DiscreteUniform{min_cells, ..., max_cells}
    min_cells: int = 10
    max_cells: int = 200

    # Number of terms in the fixed Fourier random-function expansion
    n_random_terms: int = 7

    # Scalar mediator-to-outcome coefficient
    # Chosen so that ME(t) = kappa * a(t) has a simple negative shape.
    kappa: float = -1.0

    # Observation-error standard deviations:
    # variance 0.01 corresponds to standard deviation 0.1
    sigma_x: float = 0.1
    sigma_y: float = 0.1

    # Number of grid points used to save functional truths
    n_grid: int = 501

    seed: int = 1


def validate_config(config: SimulationConfig) -> None:
    if config.n_donors <= 0:
        raise ValueError("n_donors must be positive.")

    if config.min_cells < 2:
        raise ValueError(
            "min_cells must be at least 2 so that both cell-state groups are represented."
        )

    if config.max_cells < config.min_cells:
        raise ValueError("max_cells must be at least min_cells.")

    if config.n_random_terms <= 0:
        raise ValueError("n_random_terms must be positive.")

    if config.sigma_x <= 0 or config.sigma_y <= 0:
        raise ValueError("Observation-error standard deviations must be positive.")

    if config.n_grid < 2:
        raise ValueError("n_grid must be at least 2.")


def true_functions(
    t: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """
    Return the fixed function-level truths.

    These functions do not depend on the B-spline basis dimension K
    used during model fitting.
    """
    mu_x = -0.4 + 0.5 * t + 0.25 * np.sin(2.0 * np.pi * t)

    # Exposure-to-mediator effect.
    # Together with kappa = -1, this gives
    # ME(t) = 0.05 - t - 0.65 sin(pi t):
    # slightly positive near t = 0, negative over most of the domain,
    # and mildly increasing again near t = 1.
    a_t = -0.05 + 0.65 * np.sin(np.pi * t) + 1.00 * t

    mu_y = 0.20 - 0.25 * t + 0.20 * np.cos(2.0 * np.pi * t)

    # Direct exposure effect.
    # DE(t) = -0.15 + 1.55 t + 0.48 sin(pi t):
    # slightly negative initially and then smoothly increasing.
    c_t = -0.15 + 1.55 * t + 0.48 * np.sin(np.pi * t)

    return mu_x, a_t, mu_y, c_t


def fourier_basis(
    t: FloatArray,
    n_terms: int,
) -> FloatArray:
    """
    Construct a fixed Fourier basis.

    The first column is constant. Subsequent columns are sine/cosine
    pairs. This basis is used only for generating donor-specific random
    functions and is unrelated to the fitted B-spline basis.
    """
    columns: list[FloatArray] = [np.ones_like(t, dtype=float)]
    frequency = 1

    while len(columns) < n_terms:
        columns.append(np.sqrt(2.0) * np.sin(2.0 * np.pi * frequency * t))

        if len(columns) < n_terms:
            columns.append(np.sqrt(2.0) * np.cos(2.0 * np.pi * frequency * t))

        frequency += 1

    return np.column_stack(columns)


def numerical_integral(
    values: FloatArray,
    grid: FloatArray,
) -> float:
    """Numerically integrate values over grid."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, grid))

    # Compatibility with older NumPy versions
    return float(np.trapz(values, grid))


def write_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            value,
            file,
            ensure_ascii=False,
            indent=2,
        )


def generate_dataset(
    config: SimulationConfig,
) -> dict[str, Any]:
    """Generate one complete simulation dataset."""
    validate_config(config)

    rng = np.random.default_rng(config.seed)

    # Z_i ~ N(0, 1)
    z_values = rng.normal(
        loc=0.0,
        scale=1.0,
        size=config.n_donors,
    )

    # C_i ~ DiscreteUniform{10, ..., 200}
    # rng.integers uses an exclusive upper endpoint.
    cell_counts = rng.integers(
        low=config.min_cells,
        high=config.max_cells + 1,
        size=config.n_donors,
    )

    component_index = np.arange(config.n_random_terms)

    lambda_x = 0.04 * np.power(0.45, component_index)
    lambda_y = 0.025 * np.power(0.45, component_index)

    evaluation_grid = np.linspace(
        0.0,
        1.0,
        config.n_grid,
    )

    (
        mu_x_grid,
        a_grid,
        mu_y_grid,
        c_grid,
    ) = true_functions(evaluation_grid)

    basis_on_grid = fourier_basis(
        evaluation_grid,
        config.n_random_terms,
    )

    x_data: list[list[float]] = []
    y_data: list[list[float]] = []
    t_data: list[list[float]] = []
    state_data: list[list[int]] = []

    # Noise-free trajectories evaluated at observed pseudotimes
    g_observed_grid: list[list[float]] = []
    h_observed_grid: list[list[float]] = []

    # Noise-free trajectories evaluated on a common dense grid
    g_dense_grid: list[list[float]] = []
    h_dense_grid: list[list[float]] = []

    q_coefficients_all: list[list[float]] = []
    r_coefficients_all: list[list[float]] = []

    for donor_index in range(config.n_donors):
        z_i = float(z_values[donor_index])
        c_i = int(cell_counts[donor_index])

        # Ensure that every donor contributes to both support intervals.
        n_early = c_i // 2
        n_late = c_i - n_early

        t_early = rng.uniform(
            low=0.0,
            high=0.5,
            size=n_early,
        )

        t_late = rng.uniform(
            low=0.5,
            high=1.0,
            size=n_late,
        )

        t_i = np.concatenate([t_early, t_late])

        state_i = np.concatenate(
            [
                np.zeros(n_early, dtype=int),
                np.ones(n_late, dtype=int),
            ]
        )

        # Randomize cell order so cells are not stored by state.
        permutation = rng.permutation(c_i)
        t_i = t_i[permutation]
        state_i = state_i[permutation]

        basis_at_observed_t = fourier_basis(
            t_i,
            config.n_random_terms,
        )

        q_i = rng.normal(
            loc=0.0,
            scale=np.sqrt(lambda_x),
            size=config.n_random_terms,
        )

        r_i = rng.normal(
            loc=0.0,
            scale=np.sqrt(lambda_y),
            size=config.n_random_terms,
        )

        # Donor-specific random functional deviations
        u_i_observed = basis_at_observed_t @ q_i
        v_i_observed = basis_at_observed_t @ r_i

        (
            mu_x_observed,
            a_observed,
            mu_y_observed,
            c_observed,
        ) = true_functions(t_i)

        # True mediator trajectory evaluated at observed pseudotimes
        g_i = mu_x_observed + a_observed * z_i + u_i_observed

        # True outcome trajectory evaluated at observed pseudotimes
        h_i = mu_y_observed + c_observed * z_i + config.kappa * g_i + v_i_observed

        epsilon_i = rng.normal(
            loc=0.0,
            scale=config.sigma_x,
            size=c_i,
        )

        delta_i = rng.normal(
            loc=0.0,
            scale=config.sigma_y,
            size=c_i,
        )

        x_i = g_i + epsilon_i
        y_i = h_i + delta_i

        # Evaluate this donor's true trajectories on the common grid.
        u_i_grid = basis_on_grid @ q_i
        v_i_grid = basis_on_grid @ r_i

        g_i_grid = mu_x_grid + a_grid * z_i + u_i_grid

        h_i_grid = mu_y_grid + c_grid * z_i + config.kappa * g_i_grid + v_i_grid

        x_data.append(x_i.tolist())
        y_data.append(y_i.tolist())
        t_data.append(t_i.tolist())
        state_data.append(state_i.tolist())

        g_observed_grid.append(g_i.tolist())
        h_observed_grid.append(h_i.tolist())

        g_dense_grid.append(g_i_grid.tolist())
        h_dense_grid.append(h_i_grid.tolist())

        q_coefficients_all.append(q_i.tolist())
        r_coefficients_all.append(r_i.tolist())

    me_true = config.kappa * a_grid
    de_true = c_grid
    te_true = me_true + de_true

    integrated_effects = {
        "IME": numerical_integral(me_true, evaluation_grid),
        "IDE": numerical_integral(de_true, evaluation_grid),
        "ITE": numerical_integral(te_true, evaluation_grid),
    }

    return {
        "config": asdict(config),
        "Z": z_values.tolist(),
        "C": cell_counts.tolist(),
        "X": x_data,
        "Y": y_data,
        "T": t_data,
        "cell_type": state_data,
        "g": g_observed_grid,
        "h": h_observed_grid,
        "grid": evaluation_grid.tolist(),
        "g_grid": g_dense_grid,
        "h_grid": h_dense_grid,
        "random_coefficients": {
            "q": q_coefficients_all,
            "r": r_coefficients_all,
            "lambda_x": lambda_x.tolist(),
            "lambda_y": lambda_y.tolist(),
        },
        "effect_truth": {
            "a": a_grid.tolist(),
            "c": c_grid.tolist(),
            "ME": me_true.tolist(),
            "DE": de_true.tolist(),
            "TE": te_true.tolist(),
            "kappa": config.kappa,
            "integrated_effects": integrated_effects,
        },
    }


def save_dataset(
    dataset: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    write_json(output_dir / "X.json", dataset["X"])
    write_json(output_dir / "Y.json", dataset["Y"])
    write_json(output_dir / "Z.json", dataset["Z"])
    write_json(output_dir / "C.json", dataset["C"])

    write_json(output_dir / "T.json", dataset["T"])
    write_json(
        output_dir / "cell_type.json",
        dataset["cell_type"],
    )

    # True trajectories at observed pseudotimes
    write_json(output_dir / "g.json", dataset["g"])
    write_json(output_dir / "h.json", dataset["h"])

    # Common evaluation grid and trajectory truths
    write_json(output_dir / "grid.json", dataset["grid"])
    write_json(
        output_dir / "g_grid.json",
        dataset["g_grid"],
    )
    write_json(
        output_dir / "h_grid.json",
        dataset["h_grid"],
    )

    write_json(
        output_dir / "effect_truth.json",
        dataset["effect_truth"],
    )

    write_json(
        output_dir / "random_coefficients.json",
        dataset["random_coefficients"],
    )

    write_json(
        output_dir / "config.json",
        dataset["config"],
    )

    with (output_dir / "seed.txt").open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(f"{dataset['config']['seed']}\n")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate function-level mediation simulation data "
            "independently of the fitted B-spline dimension."
        )
    )

    parser.add_argument(
        "--n",
        type=int,
        default=500,
        help="Number of donors.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory in which to save the generated dataset.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    config = SimulationConfig(
        n_donors=args.n,
        seed=args.seed,
    )

    from tempfile import TemporaryDirectory
    from dataclasses import asdict

    marker = args.output / "config.json"
    if args.output.exists():
        if (
            marker.is_file()
            and json.loads(marker.read_text()) == asdict(config)
            and (args.output / "seed.txt").is_file()
        ):
            print(f"Using generated data in {args.output}")
            return
        raise FileExistsError(f"Incomplete or different Fourier data at {args.output}.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temporary:
        stage = Path(temporary) / "generated_data"
        dataset = generate_dataset(config)
        save_dataset(dataset, stage)
        stage.rename(args.output)

    integrated = dataset["effect_truth"]["integrated_effects"]

    print(f"Saved generated dataset to: {args.output}")
    print(f"n donors: {config.n_donors}")
    print(f"C_i distribution: DiscreteUniform{{{config.min_cells},...,{config.max_cells}}}")
    print("Z_i distribution: N(0,1)")
    print(f"True IME: {integrated['IME']:.6f}")
    print(f"True IDE: {integrated['IDE']:.6f}")
    print(f"True ITE: {integrated['ITE']:.6f}")


if __name__ == "__main__":
    main()
