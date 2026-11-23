# OneK1K analysis input

The mediator is the SCTransform Pearson residual for RPS26 (`ENSG00000197728`), the outcome is the residual for BAX (`ENSG00000087088`), and the donor exposure counts rs1131017 G alleles. The study used a release covering 961 genotyped donors with 929,817 expression-profiled cells before CD4+ selection. The resulting input has 403,087 cells.

The Human Cell Atlas project describes the larger published cohort of 982 donors. Its expression matrices and cell metadata, and the GEO expression/genotyping series, provide source data. Individual-level data and the analysis-ready Seurat object are supplied by the data user. [HCA OneK1K project](https://explore.data.humancellatlas.org/projects/f2078d5f-2e7d-4844-8552-f7c41a231e52) · [GEO GSE196830](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE196830)

## From a prepared Seurat object

Run from the repository root. The `data/` directory holds your local inputs. The optional R step uses Seurat, SeuratObject, sctransform, Matrix and future. Provide an RDA containing a Seurat object named `sc` with Ensembl IDs as RNA feature names and a joint RNA counts matrix. The third and fourth arguments set the object name and the random seed.

```bash
Rscript analysis/application/preprocess_sct.R data/counts.rda data/residuals.csv sc 123
python -m analysis.application.prepare --residuals data/residuals.csv --metadata data/metadata.csv --genotypes data/genotypes.csv --g-allele ALT --output data/application.json --check-paper-sample
python -m analysis.run application
```

The R step removes zero-UMI cells and runs SCTransform before selecting CD4+ cells. It uses a 1,000-cell subsample for the model fit, requests residuals for the two genes and records the R session. If the exact study residuals are available, start with the Python preparation command. 

| File | Expected contents |
|---|---|
| `residuals.csv` | Gene IDs in the first column. Remaining column names are cell barcodes. |
| `metadata.csv` | Cell barcode in the first column, plus `donor_id` and `cell_type`. Override column names with `--barcode-column`, `--donor-column` or `--state-column`. |
| `genotypes.csv` | Variant IDs in the first column, one sample column per donor. The rs1131017 row contains diploid GT calls such as `0/0`, `0/1` and `1/1`. Phased GT and GT-first sample fields are supported. Use matching donor IDs across files. |

The study record uses REF=C and ALT=G, hence `--g-allele ALT`. If the source coding is reversed, use `--g-allele REF`. A genotype table with named `REF` and `ALT` columns allows the reader to detect the orientation automatically. Missing genotypes exclude donors and are recorded in the local preparation summary.

| Exact cell annotation | State | Interval |
|---|---|---|
| `naive thymus-derived CD4-positive, alpha-beta T cell` | 0 | [0, 0.5] |
| `central memory CD4-positive, alpha-beta T cell` | 1 | [0.5, 1] |
| `effector memory CD4-positive, alpha-beta T cell` | 1 | [0.5, 1] |

Expression is matched to metadata by barcode and to genotype by donor ID. The paper sample check expects 322, 473 and 166 donors with dosages 0, 1 and 2, respectively, and 181,285 early and 221,802 late cells. The fit, CV and summary read the same donor and cell ordering.

## From an existing analysis JSON

A donor-major JSON with `x`, `y`, `z`, `cell_type`, donor IDs and cell IDs can be used directly at `data/application.json` if `z` already counts G alleles as 0/1/2. 

```bash
python -m analysis.application.prepare --legacy-json data/old_input.json --legacy-z-offset -1 --output data/application.json --check-paper-sample
```
