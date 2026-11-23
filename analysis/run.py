"""Run the analyses for a single manuscript experiment from the repository root."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
OUTPUT = ROOT / "outputs"
SETTINGS = ANALYSIS / "settings.json"
APP_SETTINGS = ANALYSIS / "application/settings.json"
STEPS = {
    "finite": ("generate", "fit", "summarize"),
    "basis": ("generate", "fit", "cv", "summarize"),
    "aggregation": ("generate", "fit", "summarize"),
    "two_stage": ("fit", "summarize"),
    "application": ("fit", "cv", "summarize"),
    "illustrations": ("summarize",),
}


def run(module, *args, dry=False):
    argv = [sys.executable, "-m", "analysis." + module, *map(str, args)]
    print(shlex.join(argv), flush=True)
    if not dry:
        env = os.environ.copy()
        env.setdefault("MPLBACKEND", "Agg")
        subprocess.run(argv, cwd=ROOT, check=True, env=env)


def paper_fit(data, output, k, settings, seed, replicate, dry, *, oracle=False, application=False):
    source = "--application-data" if application else "--data"
    extra = ["--replicate", replicate] if replicate is not None else []
    if oracle:
        extra.append("--oracle")
    run(
        "fit_model",
        source,
        data,
        "--output",
        output,
        "--K",
        k,
        "--seed",
        seed,
        "--iterations",
        settings["iterations"],
        "--burnin",
        settings["burnin"],
        "--aux-thin",
        settings["aux_thin"],
        "--resume",
        *extra,
        dry=dry,
    )


def simulation(exp, step, cfg, reps, out, cv_out, dry):
    design = cfg[exp]
    base = OUTPUT / exp
    if step == "generate":
        if exp == "basis":
            for rep in reps:
                run(
                    "simulation.generate_basis",
                    "--n",
                    design["n"],
                    "--seed",
                    design["data_seed_base"] + rep,
                    "--output",
                    base / f"replicate_{rep:03d}" / "generated_data",
                    dry=dry,
                )
        else:
            run("simulation.generate_spline", exp, "--replicates", *reps, dry=dry)
    elif step == "fit":
        for rep in reps:
            repdir = base / f"replicate_{rep:03d}"
            if exp == "basis":
                for k in design["Ks"]:
                    paper_fit(
                        repdir / "generated_data",
                        repdir / f"fit_K_{k}",
                        k,
                        cfg["simulation"],
                        design["mcmc_seed_base"] + 10 * rep + k,
                        rep,
                        dry,
                    )
                continue
            sizes = design["sample_sizes"] if exp == "finite" else [design["n"]]
            for n in sizes:
                k = design["K"]
                folder = repdir / f"n_{n}"
                data = folder / "generated_data"
                seed = design["mcmc_seed_base"] + 2 * rep + n
                paper_fit(data, folder / f"fit_K{k}", k, cfg["simulation"], seed, rep, dry)
                if exp == "aggregation":
                    paper_fit(
                        data,
                        folder / f"fit_K{k}_oracleT",
                        k,
                        cfg["simulation"],
                        seed,
                        rep,
                        dry,
                        oracle=True,
                    )
                    run(
                        "simulation.fit_donor_average_scalar",
                        "--data",
                        data,
                        "--output",
                        folder / "fit_scalar",
                        "--bootstrap",
                        design["bootstrap"],
                        "--seed",
                        design["bootstrap_seed_base"] + rep,
                        "--resume",
                        dry=dry,
                    )
    elif step == "cv":
        run(
            "simulation.cv_basis",
            "--root",
            base,
            "--output",
            cv_out,
            "--replicates",
            *reps,
            "--Ks",
            *design["Ks"],
            "--n-folds",
            design["n_folds"],
            "--resume",
            "--restart-incomplete",
            dry=dry,
        )
    else:
        shared = [
            "--root",
            base,
            "--replicates",
            *reps,
            "--expected-replicates",
            len(reps),
            "--output-dir",
            out,
        ]
        if exp == "finite":
            run(
                "simulation.evaluate_finite",
                *shared,
                "--sample-sizes",
                *design["sample_sizes"],
                "--fit-dir-name",
                f"fit_K{design['K']}",
                dry=dry,
            )
            run("tables", "finite", "--summary", out, "--output", out, dry=dry)
        elif exp == "basis":
            run("simulation.evaluate_basis", *shared, "--K-values", *design["Ks"], dry=dry)
            run("tables", "basis", "--summary", out, "--cv", cv_out, "--output", out, dry=dry)
        else:
            run(
                "simulation.evaluate_aggregation",
                *shared,
                "--n",
                design["n"],
                "--K",
                design["K"],
                dry=dry,
            )
            run(
                "simulation.plot_aggregation",
                "--input-csv",
                out / "aggregation_effect_curves_pointwise.csv",
                "--output-dir",
                out,
                dry=dry,
            )


def two_stage(step, cfg, reps, out, dry):
    design = cfg["two_stage"]
    base = OUTPUT / "two_stage"
    if step == "summarize":
        run(
            "simulation.evaluate_two_stage_comparison",
            "--root",
            OUTPUT / "finite",
            "--output-root",
            base,
            "--summary-dir",
            out,
            "--replicates",
            *reps,
            dry=dry,
        )
        run(
            "simulation.plot_two_stage",
            "--input-csv",
            out / "effect_curve_summary.csv",
            "--output-dir",
            out,
            dry=dry,
        )
        return
    # Generate the same nested master dataset as Table 1. Only the shared n=1000
    # joint fit is needed for this comparison.
    run("simulation.generate_spline", "finite", "--replicates", *reps, dry=dry)
    for rep in reps:
        n, k = design["n"], design["K"]
        folder = OUTPUT / "finite" / f"replicate_{rep:03d}" / f"n_{n}"
        seed = cfg["finite"]["mcmc_seed_base"] + 2 * rep + n
        paper_fit(
            folder / "generated_data", folder / f"fit_K{k}", k, cfg["simulation"], seed, rep, dry
        )
        stage_options = []
        for key in (
            "mu_alpha_sd",
            "mu_beta_sd",
            "var_alpha_shape",
            "var_alpha_scale",
            "var_beta_shape",
            "var_beta_scale",
        ):
            stage_options.extend(["--stage1-" + key.replace("_", "-"), design[key]])
        run(
            "simulation.fit_two_stage_comparison",
            "--root",
            OUTPUT / "finite",
            "--output-root",
            base,
            "--replicate",
            rep,
            "--iterations",
            cfg["simulation"]["iterations"],
            "--burnin",
            cfg["simulation"]["burnin"],
            "--seed",
            design["seed"],
            "--joint-source",
            "reuse",
            *stage_options,
            dry=dry,
        )


def application(step, cfg, dry):
    from .application.cv import read_config
    from .application.prepare import check_paper_sample, validate

    settings = cfg["application"]
    app = read_config(APP_SETTINGS)
    if app["candidate_Ks"] != settings["Ks"] or app["hyperparameters"] != cfg["hyperparameters"]:
        raise ValueError(
            "OneK1K and simulation settings disagree on candidate dimensions or priors."
        )
    if any(app[k] != settings[k] for k in ("iterations", "burnin")) or app["z_offset"] != 0:
        raise ValueError(
            "Check OneK1K iterations, burn-in and dosage encoding in the two settings files."
        )
    if not dry:
        if not app["data_json"].is_file():
            raise FileNotFoundError(
                "Prepare data/application.json as described in analysis/application/README.md."
            )
        check_paper_sample(validate(json.loads(app["data_json"].read_text(encoding="utf-8"))))
    if step == "fit":
        for k in settings["Ks"]:
            paper_fit(
                app["data_json"],
                app["posterior_files"][str(k)].parent,
                k,
                settings,
                settings["seed"],
                None,
                dry,
                application=True,
            )
    elif step == "cv":
        run("application.cv", "--config", APP_SETTINGS, "--resume", "--restart-incomplete", dry=dry)
    else:
        run(
            "application.summarize",
            "--root",
            APP_SETTINGS.parent,
            "--config",
            APP_SETTINGS.name,
            "--output",
            OUTPUT / "application/summary",
            "--overwrite",
            "--rho-iterations",
            settings["iterations"],
            "--rho-burnin",
            settings["burnin"],
            "--rho-seed",
            settings["rho_seed"],
            dry=dry,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=STEPS)
    parser.add_argument(
        "--step", choices=["all", "generate", "fit", "cv", "summarize"], default="all"
    )
    parser.add_argument(
        "--replicates", type=int, nargs="+", help="Simulation replicate IDs (paper default 1–10)."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without fitting or writing results."
    )
    args = parser.parse_args()
    cfg = json.loads(SETTINGS.read_text())
    if args.replicates and args.experiment in ("illustrations", "application"):
        parser.error("--replicates applies to simulations.")
    reps = sorted(cfg["replicates"] if args.replicates is None else args.replicates)
    if len(reps) != len(set(reps)) or not set(reps) <= set(cfg["replicates"]):
        parser.error("Use distinct replicate IDs from 1 to 10.")
    exp = args.experiment
    steps = STEPS[exp] if args.step == "all" else (args.step,)
    if not set(steps) <= set(STEPS[exp]):
        parser.error(f"{exp} supports {', '.join(STEPS[exp])}.")
    base = OUTPUT / exp
    # A subset summary has its own destination and cannot overwrite a full-study table.
    summary_base = (
        base if reps == cfg["replicates"] else base / "subsets" / ("_".join(map(str, reps)))
    )
    for step in steps:
        if exp == "illustrations":
            run("illustrations", "--output", base, dry=args.dry_run)
        elif exp == "application":
            application(step, cfg, args.dry_run)
        elif exp == "two_stage":
            two_stage(step, cfg, reps, summary_base / "summary", args.dry_run)
        else:
            simulation(
                exp, step, cfg, reps, summary_base / "summary", summary_base / "cv", args.dry_run
            )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        raise SystemExit(str(exc))
