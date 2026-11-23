# BLaFM

Bayesian latent functional mediation for donor-grouped single-cell measurements. This repository accompanies *Bayesian Latent Functional Mediation Analysis with Application to Population-Scale Single-Cell Data* by Zehua Li, Yang Ni and Weixuan Zhu.

BLaFM estimates how the mediated, direct and total effects of a unit change in a donor-level exposure vary across a shared latent cell state. A cubic B-spline represents each donor's mediator and outcome trajectories. Cell annotations constrain the state interval, and the joint sampler updates the cell states and donor trajectories together.

## Run the example at K = 5

From the repository root, with Python 3.10 or later, run

```bash
python -m pip install ".[paper]"
python example.py --plot
```

This runs the first replicate of *Sensitivity to functional basis complexity*. It generates 500 donors with 10–200 cells each. Analytic functions specify the effect curves, and a seven-term Fourier expansion generates donor-specific deviations. The joint model estimates the latent cell states and fits the trajectories with five cubic B-spline basis functions. The default chain runs for 10,000 iterations and discards the first 5,000.

Open `outputs/example/effects.pdf` to see the mediated, direct and total effect curves with their generating truths, posterior means and pointwise 95% credible intervals. The directory also contains the four observed inputs in `data.json` and curve summaries at 501 grid points in `effects.csv`. The terminal prints the estimates alongside the integrated truths (IME −0.8638, IDE 0.9306, ITE 0.0668) and the root integrated squared error (RISE) of each curve, multiplied by 100. 

For a quick check with the same observations and K = 5, shorten the chain and use a separate output directory.

```bash
python example.py --plot --iterations 200 --burnin 100 --out outputs/example_short
```

For the model and the CSV example alone, install with `python -m pip install .` and run `python example.py`.

## Fit real data

```python
from blafm import fit, summarize_effects

posterior = fit(x, y, z, cell_type, K=5, iterations=10_000, burnin=5_000, seed=123)
effects = summarize_effects(posterior)
print(effects["IME"], effects["IDE"], effects["ITE"])
```

| Input | Meaning |
|---|---|
| `x[i]`, `y[i]` | Mediator and outcome for the same ordered cells of donor `i`. Rows may have different lengths across donors. |
| `z[i]` | One finite numeric exposure per donor. |
| `cell_type[i]` | One state label per cell. Label 0 restricts pseudotime to [0, 0.5] and label 1 to [0.5, 1]. |
| `K` | Total number of cubic basis functions, including boundary basis functions. The minimum is four. |

The command line accepts the same four fields in JSON, with optional donor and cell IDs. For example, after running the example above,

```bash
python -m blafm outputs/example/data.json --K 5 --iterations 10000 --burnin 5000 --seed 100015 --out outputs/example/fit.npz
```

`fit.npz` includes the retained draws of `alpha0`, `A`, `beta0`, `B_diag` and `C`, donor coefficient means, cell-state means and the effect summaries. The flat array `t_mean` follows the donor/cell input order. Use `cell_counts` to split it by donor. `ME`, `DE` and `TE` are evaluated on `grid`. Their integrated counterparts are `IME`, `IDE` and `ITE`. Each summary uses the column order `mean, sd, q025, q975`.

## Paper analyses

The [analysis guide](analysis/README.md) gives the commands, study settings, table and figure outputs, and the status of numerical verification. The [OneK1K input guide](analysis/application/README.md) identifies the data sources and explains how to prepare the study cohort.

| File or directory | Purpose |
|---|---|
| `example.py` | Generate one basis-complexity replicate, fit the K = 5 model and compare the three effect curves with their known values. |
| `blafm/model.py`, `blafm/mcmc.py` | Joint sampler and its conditional updates. The example and the paper full fits share this code. |
| `blafm/effects.py` | Calculate effects within matched posterior draws and integrate over pseudotime. |
| `analysis/run.py` | Run simulation, the OneK1K analysis, or the model illustrations. |
| `tests/` | Numerical checks for the model, paper export and data preparation. With the paper dependencies installed, run `python -m unittest discover -s tests`. |
