"""Fast numerical checks for the public BLaFM interface."""

import unittest
import numpy as np
from blafm import fit, summarize_effects
from blafm.mcmc import cubic_basis


class ModelChecks(unittest.TestCase):
    def test_ragged_states_draws_and_seed(self):
        data = dict(
            x=[[0, 1, 2], [0.4]], y=[[0, 0.4, -1], [0.3]], z=[0, 1], cell_type=[[0, 0, 1], [1]]
        )
        options = dict(K=4, iterations=14, burnin=6, seed=921)
        first, second = fit(**data, **options), fit(**data, **options)
        for key in ["A", "B_diag", "C", "t_mean", "alpha_mean", "beta_mean"]:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(first["A"].shape, (8, 4))
        np.testing.assert_array_equal(first["cell_counts"], [3, 1])
        lower = 0.5 * np.concatenate(data["cell_type"])
        self.assertTrue(np.all((lower <= first["t_mean"]) & (first["t_mean"] <= lower + 0.5)))
        bad = {**data, "cell_type": [[0, 2, 1], [1]]}
        with self.assertRaises(ValueError):
            fit(**bad, **options)

    def test_exact_integrals_and_matched_draws(self):
        a = np.broadcast_to([[1.0], [3.0]], (2, 4))
        b = np.broadcast_to([[2.0], [4.0]], (2, 4))
        c = np.broadcast_to([[0.2], [0.4]], (2, 4))
        result = summarize_effects({"A": a, "B_diag": b, "C": c}, [0, 0.5, 1])
        np.testing.assert_allclose(result["ME"][:, 0], 7)
        np.testing.assert_allclose(result["IME"][0], 7)
        np.testing.assert_allclose(result["TE"][:, 0], result["ME"][:, 0] + result["DE"][:, 0])
        np.testing.assert_allclose(result["ITE"][0], 7.3)
        for k in (4, 5, 7):
            np.testing.assert_allclose(sum(cubic_basis(k).integrate(0, 1)), 1)

    def test_one_donor_and_local_random_streams(self):
        saved = np.random.get_state()
        result = fit([[1]], [[0]], [0], [[1]], K=4, iterations=4, burnin=2, seed=0)
        np.testing.assert_array_equal(saved[1], np.random.get_state()[1])
        self.assertEqual(saved[2:], np.random.get_state()[2:])
        self.assertTrue(np.isfinite(result["B_diag"]).all())


if __name__ == "__main__":
    unittest.main()
