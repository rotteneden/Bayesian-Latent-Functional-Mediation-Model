"""Match expression residuals, cell annotations and rs1131017 genotypes by identifier."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.fit_model import atomic_json, read_json, digest

X_GENE = "ENSG00000197728"
Y_GENE = "ENSG00000087088"
STATES = {
    "naive thymus-derived CD4-positive, alpha-beta T cell": 0,
    "central memory CD4-positive, alpha-beta T cell": 1,
    "effector memory CD4-positive, alpha-beta T cell": 1,
}
GENOTYPES = {"0/0": 0, "0/1": 1, "1/0": 1, "1/1": 2, "0|0": 0, "0|1": 1, "1|0": 1, "1|1": 2}


def validate(doc):
    n = len(doc["x"])
    if not n or any(len(doc[k]) != n for k in ("y", "z", "cell_type", "donor_ids", "cell_ids")):
        raise ValueError("Invalid donor counts.")
    if len(set(doc["donor_ids"])) != n:
        raise ValueError("Duplicate donor IDs.")
    nc = []
    states = []
    for i in range(n):
        c = len(doc["x"][i])
        nc.append(c)
        if c == 0 or any(len(doc[k][i]) != c for k in ("y", "cell_type", "cell_ids")):
            raise ValueError("Empty donor or inconsistent row lengths.")
        if len(set(doc["cell_ids"][i])) != c:
            raise ValueError("Duplicate cell IDs.")
        for k in ("x", "y"):
            if not np.isfinite(doc[k][i]).all():
                raise ValueError("Nonfinite expression.")
        if not np.isin(doc["cell_type"][i], [0, 1]).all():
            raise ValueError("Invalid cell state.")
        states.extend(doc["cell_type"][i])
    z = np.asarray(doc["z"])
    if z.shape != (n,) or not np.isin(z, [0, 1, 2]).all():
        raise ValueError("Invalid G-allele dosage.")
    return {
        "n_donors": n,
        "n_cells": sum(nc),
        "genotype_counts": np.bincount(z.astype(int), minlength=3).tolist(),
        "state_counts": np.bincount(np.asarray(states, dtype=int), minlength=2).tolist(),
        "cells_per_donor_median": float(np.median(nc)),
        "cells_per_donor_min": min(nc),
        "cells_per_donor_max": max(nc),
    }


def check_paper_sample(summary):
    expected = {
        "n_donors": 961,
        "n_cells": 403087,
        "genotype_counts": [322, 473, 166],
        "state_counts": [181285, 221802],
        "cells_per_donor_median": 414.0,
        "cells_per_donor_min": 19,
        "cells_per_donor_max": 1031,
    }
    wrong = {
        k: {"observed": summary[k], "paper": v} for k, v in expected.items() if summary[k] != v
    }
    if wrong:
        raise ValueError("Sample differs from manuscript: " + json.dumps(wrong))


def prepare(args):
    out = Path(args.output)
    if out.exists() and not args.overwrite:
        raise FileExistsError(out)
    provenance = {}
    if args.legacy_json:
        if args.legacy_z_offset is None:
            raise ValueError("Specify --legacy-z-offset explicitly: archived 1/2/3 JSON uses -1.")
        raw = read_json(args.legacy_json)
        doc = {k: raw[k] for k in ("x", "y", "cell_type")}
        z = np.asarray(raw["z"], float).reshape(-1) + args.legacy_z_offset
        if not np.isin(z, [0, 1, 2]).all():
            raise ValueError("Invalid legacy dosage after offset.")
        doc["z"] = z.astype(int).tolist()
        ids = raw.get("donor_ids", raw.get("donor_id"))
        if ids is None:
            raise ValueError("Legacy JSON lacks donor identity.")
        doc["donor_ids"] = [str(x) for x in ids]
        if "cell_ids" in raw:
            doc["cell_ids"] = raw["cell_ids"]
            identity = "original_barcodes"
        else:
            # These labels preserve row identity, and must never be described as barcodes.
            doc["cell_ids"] = [
                [f"legacy_row:{d}:{c}" for c in range(len(x))]
                for d, x in zip(doc["donor_ids"], doc["x"])
            ]
            identity = "synthetic_legacy_row_labels_not_barcodes"
        provenance = {
            "legacy_source_sha256": digest(args.legacy_json),
            "legacy_z_offset": args.legacy_z_offset,
            "cell_identity": identity,
        }
    else:
        if not all((args.residuals, args.metadata, args.genotypes)):
            raise ValueError("Provide residuals, metadata and genotypes, or --legacy-json.")
        residual = pd.read_csv(args.residuals, index_col=0)
        if not residual.index.is_unique or not residual.columns.is_unique:
            raise ValueError("Residual matrix has duplicate gene/cell labels.")
        residual = residual.loc[[X_GENE, Y_GENE]].T.rename(columns={X_GENE: "x", Y_GENE: "y"})
        residual.index = residual.index.astype(str)
        residual.index.name = "cell_id"
        meta = pd.read_csv(args.metadata, dtype=str)
        barcode = args.barcode_column or meta.columns[0]
        meta = meta.rename(
            columns={
                barcode: "cell_id",
                args.donor_column: "donor_id",
                args.state_column: "annotation",
            }
        )
        if meta.cell_id.duplicated().any():
            raise ValueError("Duplicate metadata barcodes.")
        combined = residual.reset_index().merge(
            meta[["cell_id", "donor_id", "annotation"]],
            on="cell_id",
            how="left",
            sort=False,
            validate="one_to_one",
        )
        if combined[["donor_id", "annotation"]].isna().any().any():
            raise ValueError("Expression barcodes missing donor/state metadata.")
        combined["state"] = combined.annotation.map(STATES)
        combined = combined[combined.state.notna()].copy()
        genotype = pd.read_csv(args.genotypes, index_col=0, dtype=str)
        if not genotype.index.is_unique or not genotype.columns.is_unique:
            raise ValueError("Duplicate genotype variant/donor labels.")
        raw = genotype.loc["rs1131017"]
        g_allele = args.g_allele
        if ("REF" in raw.index) != ("ALT" in raw.index):
            raise ValueError("Provide both REF and ALT columns when recording allele labels.")
        if "REF" in raw.index and "ALT" in raw.index:
            alleles = [str(raw["REF"]).upper(), str(raw["ALT"]).upper()]
            if set(alleles) != {"C", "G"}:
                raise ValueError("Expected a biallelic C/G record for rs1131017.")
            resolved = "ALT" if alleles[1] == "G" else "REF"
            if g_allele is not None and g_allele != resolved:
                raise ValueError(
                    "The specified G allele conflicts with the rs1131017 REF/ALT record."
                )
            g_allele = resolved
        elif g_allele is None:
            raise ValueError(
                "Supply REF and ALT columns or specify --g-allele ALT or REF from the source record."
            )
        raw = raw.drop(labels=["REF", "ALT"], errors="ignore")
        gt = raw.map(lambda x: str(x).split(":")[0]).map(GENOTYPES)
        if g_allele == "REF":
            gt = 2 - gt
        doc = {k: [] for k in ("x", "y", "z", "cell_type", "donor_ids", "cell_ids")}
        missing = []
        for donor, frame in combined.groupby("donor_id", sort=False):
            if donor not in gt or pd.isna(gt[donor]):
                missing.append(str(donor))
                continue
            doc["donor_ids"].append(str(donor))
            doc["cell_ids"].append(frame.cell_id.astype(str).tolist())
            doc["x"].append(pd.to_numeric(frame.x, errors="raise").tolist())
            doc["y"].append(pd.to_numeric(frame.y, errors="raise").tolist())
            doc["cell_type"].append(frame.state.astype(int).tolist())
            doc["z"].append(int(gt[donor]))
        provenance = {
            "source_sha256": {
                str(p): digest(p) for p in (args.residuals, args.metadata, args.genotypes)
            },
            "g_allele_field": g_allele,
            "excluded_missing_genotype_donors": missing,
            "cell_identity": "original_barcodes",
        }
    summary = validate(doc)
    if args.check_paper_sample:
        check_paper_sample(summary)
    doc["provenance"] = provenance
    doc["encoding"] = {
        "z": "rs1131017 G dosage 0/1/2",
        "state_mapping": STATES,
        "expression_scale": "SCTransform Pearson residuals",
        "z_offset": 0,
    }
    atomic_json(out, doc)
    atomic_json(
        out.with_suffix(".summary.json"),
        dict(summary=summary, provenance=provenance, data_sha256=digest(out)),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("residuals", "metadata", "genotypes", "legacy-json"):
        p.add_argument("--" + flag, type=Path)
    p.add_argument(
        "--g-allele",
        choices=["REF", "ALT"],
        help="Source allele encoding when REF/ALT columns are absent.",
    )
    p.add_argument("--legacy-z-offset", type=float)
    p.add_argument("--barcode-column")
    p.add_argument("--donor-column", default="donor_id")
    p.add_argument("--state-column", default="cell_type")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--check-paper-sample", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    prepare(p.parse_args())
