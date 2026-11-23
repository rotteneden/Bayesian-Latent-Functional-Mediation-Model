"""Check the manuscript example, posterior export and application allele convention."""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from analysis.application import prepare
from analysis.fit_model import fit as write_paper_fit
from analysis.simulation.generate_basis import SimulationConfig, generate_dataset
from blafm import fit, summarize_effects

SETTINGS = Path(__file__).resolve().parents[1] / "analysis" / "settings.json"


class PaperChecks(unittest.TestCase):
    def test_basis_example_uses_paper_data_and_k5_fit(self):
        root = SETTINGS.parents[1]
        with TemporaryDirectory() as directory:
            out = Path(directory)
            subprocess.run(
                [sys.executable, str(root / "example.py"), "--iterations", "4", "--out", directory],
                check=True, capture_output=True, text=True,
            )
            observed = json.loads((out / "data.json").read_text(encoding="utf-8"))
            rows = np.genfromtxt(
                out / "effects.csv", names=True, dtype=None, delimiter=",", encoding="utf-8"
            )

        reference = generate_dataset(SimulationConfig(n_donors=500, seed=1001))
        for key, source in (("x", "X"), ("y", "Y"), ("z", "Z"), ("cell_type", "cell_type")):
            self.assertEqual(observed[key], reference[source])
        draws = fit(**observed, K=5, seed=100015, iterations=4, burnin=2)
        expected = summarize_effects(draws, reference["grid"])
        for key in ("ME", "DE", "TE"):
            curve = rows[rows["effect"] == key]
            np.testing.assert_array_equal(curve["t"], reference["grid"])
            np.testing.assert_array_equal(curve["truth"], reference["effect_truth"][key])
            for index, column in enumerate(expected["summary_columns"]):
                np.testing.assert_allclose(curve[column], expected[key][:, index])

    def test_public_and_manuscript_draws(self):
        fields = dict(
            x=[[0.1, 0.9], [0.2]],
            y=[[0.2, -0.3], [-0.8]],
            z=[1, 0],
            cell_type=[[0, 1], [0]],
            donor_ids=["d1", "d2"],
            cell_ids=[["a", "b"], ["c"]],
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.json"
            source.write_text(json.dumps(fields))
            args = SimpleNamespace(
                data=None,
                application_data=source,
                output=root / "fit",
                K=4,
                iterations=10,
                burnin=4,
                aux_thin=2,
                seed=34,
                oracle=False,
                quiet=True,
                paper_config=SETTINGS,
                z_offset=0,
                replicate=None,
                overwrite=False,
            )
            archive, _ = write_paper_fit(args)
            public = fit(
                **{k: fields[k] for k in ("x", "y", "z", "cell_type")},
                K=4,
                iterations=10,
                burnin=4,
                seed=34,
            )
            np.testing.assert_array_equal(archive["aux_saved_itr"], [5, 7, 9])
            self.assertEqual(len(archive["A"]), 11)
            np.testing.assert_array_equal(np.array(archive["A"])[5:, 0, :], public["A"])
            np.testing.assert_array_equal(
                np.diagonal(np.array(archive["B"])[5:], axis1=1, axis2=2), public["B_diag"]
            )
            args.resume = True
            self.assertIsNone(write_paper_fit(args)[0])
            args.seed += 1
            with self.assertRaises(FileExistsError):
                write_paper_fit(args)

    def test_allele_orientation_and_barcode_alignment(self):
        with TemporaryDirectory() as directory:
            d = Path(directory)
            (d / "res.csv").write_text(
                "gene,cell2,cell1\nENSG00000197728,2,1\nENSG00000087088,4,3\n"
            )
            early, late = list(prepare.STATES)[:2]
            (d / "meta.csv").write_text(
                f'barcode,donor_id,cell_type\ncell1,D1,"{early}"\ncell2,D2,"{late}"\n'
            )
            (d / "gt.csv").write_text("id,REF,ALT,D1,D2\nrs1131017,G,C,1/1,0/0\n")
            args = SimpleNamespace(
                residuals=d / "res.csv",
                metadata=d / "meta.csv",
                genotypes=d / "gt.csv",
                output=d / "input.json",
                legacy_json=None,
                g_allele=None,
                barcode_column="barcode",
                donor_column="donor_id",
                state_column="cell_type",
                check_paper_sample=False,
                overwrite=False,
            )
            prepare.prepare(args)
            doc = json.loads(args.output.read_text())
            self.assertEqual(doc["donor_ids"], ["D2", "D1"])
            self.assertEqual(doc["z"], [2, 0])
            self.assertEqual(doc["cell_type"], [[1], [0]])
            self.assertEqual(doc["y"], [[4], [3]])
            args.g_allele = "ALT"
            args.overwrite = True
            with self.assertRaisesRegex(ValueError, "conflicts"):
                prepare.prepare(args)


if __name__ == "__main__":
    unittest.main()
