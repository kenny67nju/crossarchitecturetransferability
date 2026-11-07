"""Regression checks for the published scoring definitions and supports."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from metrics import effective_rank, effective_rank_from_singular_values, compute_subspace_alignment, ssr_dagger
from final_selection import multi, selections


class FinalMetricsTest(unittest.TestCase):
    def test_rankme_uses_singular_values(self):
        sigma = np.array([4., 2., 1.])
        p = sigma / sigma.sum()
        expected = np.exp(-(p * np.log(p)).sum())
        self.assertAlmostEqual(effective_rank_from_singular_values(sigma), expected)
        self.assertAlmostEqual(effective_rank(np.diag(sigma)), expected)

    def test_joint_rotation_invariance(self):
        rng = np.random.default_rng(42)
        a, b = rng.normal(size=(21, 8)), rng.normal(size=(18, 8))
        q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        sa = compute_subspace_alignment(a, b, k=3)['sa_k']
        rotated = compute_subspace_alignment(a @ q, b @ q, k=3)['sa_k']
        self.assertAlmostEqual(sa, rotated)
        self.assertAlmostEqual(ssr_dagger(effective_rank(a), sa, 10),
                               ssr_dagger(effective_rank(a @ q), rotated, 10))

    def test_dimension_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            compute_subspace_alignment(np.ones((5, 3)), np.ones((5, 4)))

    def test_ssr_formula_and_clip(self):
        self.assertAlmostEqual(ssr_dagger(4., .25, 10), (np.log(4) - np.log(.25)) / np.log(10))
        self.assertAlmostEqual(ssr_dagger(4., 0., 10), ssr_dagger(4., 1e-12, 10))
        with self.assertRaises(ValueError):
            ssr_dagger(4., .25, 1)

    def test_final_mv_support_and_selection(self):
        folder = ROOT / 'results/p0_vgg_pair_20261104/p0_aggregate_v2'
        frame = multi(pd.read_csv(folder / 'single_preprocess_4096_pool132_scores.csv'))
        actual = selections(frame)
        expected = pd.read_csv(folder / 'single_preprocess_4096_multi119_selection.csv')
        for method in ['rankme', 'ssr_dagger', 'random_exact', 'disco_official_g6', 'disco_official_g8']:
            a = actual[actual.method == method].sort_values('task')
            e = expected[expected.method == method].sort_values('task')
            np.testing.assert_allclose(a.lp_acc, e.lp_acc, atol=1e-12)
            np.testing.assert_allclose(a.regret, e.regret, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
