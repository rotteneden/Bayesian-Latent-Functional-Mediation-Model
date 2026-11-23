"""Fit the K=5 model in the paper's functional basis complexity experiment."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analysis.simulation.generate_basis import SimulationConfig, generate_dataset, numerical_integral
from blafm import fit, summarize_effects


K = 5
SETTINGS = Path(__file__).resolve().parent / "analysis" / "settings.json"
EFFECT_LABELS = {"ME": "Mediated", "DE": "Direct", "TE": "Total"}


def simulate(config):
    dataset = generate_dataset(config)
    observations = {
        "x": dataset["X"],
        "y": dataset["Y"],
        "z": dataset["Z"],
        "cell_type": dataset["cell_type"],
    }
    return observations, np.asarray(dataset["grid"]), dataset["effect_truth"]


def main():
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    design = settings["basis"]
    chain = settings["simulation"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replicate", type=int, choices=settings["replicates"], default=1,
        help="Manuscript simulation replicate (default 1).",
    )
    parser.add_argument(
        "--iterations", type=int, default=chain["iterations"],
        help=f"Number of MCMC iterations (default {chain['iterations']}).",
    )
    parser.add_argument(
        "--burnin", type=int, help="Manuscript burn-in by default, or half a custom chain."
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Write a PDF of the three effect curves (requires matplotlib).",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/example"))
    args = parser.parse_args()
    burnin = args.burnin
    if burnin is None:
        burnin = chain["burnin"] if args.iterations == chain["iterations"] else args.iterations // 2
    if not 0 <= burnin <= args.iterations - 2:
        parser.error("Burn-in must leave at least two posterior draws.")
    if args.plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            parser.error(f'To use --plot, run python -m pip install ".[paper]". {exc}')
    args.out.mkdir(parents=True, exist_ok=True)

    data_seed = design["data_seed_base"] + args.replicate
    fit_seed = design["mcmc_seed_base"] + 10 * args.replicate + K
    config = SimulationConfig(n_donors=design["n"], seed=data_seed)
    data, grid, truth = simulate(config)
    print(
        f"Basis complexity, replicate {args.replicate}: {config.n_donors} donors, "
        f"{sum(map(len, data['x']))} cells, K={K}, data seed={data_seed}, fit seed={fit_seed}.\n"
        f"Running {args.iterations} iterations with {burnin} burn-in.",
        flush=True,
    )
    posterior = fit(
        **data, K=K, iterations=args.iterations, burnin=burnin, seed=fit_seed, progress=True
    )
    effects = summarize_effects(posterior, grid=grid)

    (args.out / "data.json").write_text(json.dumps(data), encoding="utf-8")
    with (args.out / "effects.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["effect", "t", "truth", *effects["summary_columns"]])
        for name in EFFECT_LABELS:
            for t, target, row in zip(grid, truth[name], effects[name]):
                writer.writerow([name, t, target, *row])

    print("Integrated effect    Truth      Mean      2.5%     97.5%")
    for name in EFFECT_LABELS:
        mean, _, lo, hi = effects["I" + name]
        target = truth["integrated_effects"]["I" + name]
        print(f"{'I' + name:17} {target:7.4f}   {mean:7.4f}   {lo:7.4f}   {hi:7.4f}")

    print("Effect-function RISE x 100 for this replicate")
    for name in EFFECT_LABELS:
        squared_error = (effects[name][:, 0] - truth[name]) ** 2
        print(f"{name}: {100 * np.sqrt(numerical_integral(squared_error, grid)):.3f}")

    if args.plot:
        fig, axes = plt.subplots(
            1, 3, figsize=(9.4, 3.1), sharex=True, sharey=True, layout="constrained"
        )
        for ax, (key, label) in zip(axes, EFFECT_LABELS.items()):
            mean, _, lo, hi = effects[key].T
            ax.fill_between(grid, lo, hi, color="#6195ba", alpha=0.25, label="Pointwise 95% CrI")
            ax.plot(grid, mean, color="#22597d", label="Posterior mean")
            ax.plot(grid, truth[key], "--", color="#272727", label="Truth")
            ax.set(xlabel="Latent cell state", title=label)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Effect")
        axes[0].legend(frameon=False, fontsize=7)
        fig.savefig(args.out / "effects.pdf")
        plt.close(fig)
    print(f"Saved example files in {args.out}")


if __name__ == "__main__":
    main()
