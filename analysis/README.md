# Analyses

Run commands from the BLaFM repository root after installing the analysis dependencies with `python -m pip install ".[paper]"`. For a short introduction to fitting and plotting, run `python example.py --plot`.

## Run an analysis

Choose one command. Simulations generate their own data from fixed seeds. The OneK1K analysis requires prepared study data. [OneK1K input guide](application/README.md)

| Paper output | Command | Results under `outputs/` |
|---|---|---|
| Fig. 1 and Fig. S1 · Model illustrations | `python -m analysis.run illustrations` | `illustrations/` |
| Table 1 · Finite-sample recovery | `python -m analysis.run finite` | `finite/summary/table_1_finite.csv` |
| Table 2 · Sensitivity to functional basis complexity | `python -m analysis.run basis` | `basis/summary/table_2_basis.csv` |
| Table 3 and Fig. 2 · Aggregation | `python -m analysis.run aggregation` | `aggregation/summary/` |
| Table 4 and Fig. 3 · Two-stage comparison | `python -m analysis.run two_stage` | `two_stage/summary/` |
| Tables 5–6 and Figs. 4–5 · OneK1K | `python -m analysis.run application` | `application/summary/main/` and `application/summary/supplement/` |

To preview the basis-complexity workflow for replicate 1, run the first command below. The second executes it for all candidate dimensions. Subset summaries are written under `outputs/basis/subsets/`.

```bash
python -m analysis.run basis --replicates 1 --dry-run
python -m analysis.run basis --replicates 1
```
