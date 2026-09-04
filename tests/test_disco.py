import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from metrics.disco import disco_classification


class TestDisco(unittest.TestCase):
    def test_spectral_ratios_cover_the_full_spectrum(self):
        rng = np.random.default_rng(0)
        features = rng.normal(size=(24, 7))
        labels = np.repeat(np.arange(3), 8)
        details = disco_classification(
            features, labels, n_groups=4, pca_components=0, return_details=True
        )
        self.assertTrue(np.isclose(details.spectral_ratios.sum(), 1.0))
        self.assertEqual(details.group_bounds, ((0, 2), (2, 4), (4, 6), (6, 7)))
        self.assertLessEqual(0.0, details.score)
        self.assertLessEqual(details.score, 1.0)

    def test_score_is_invariant_to_right_orthogonal_rotation(self):
        rng = np.random.default_rng(1)
        features = rng.normal(size=(30, 9))
        labels = np.repeat(np.arange(3), 10)
        rotation, _ = np.linalg.qr(rng.normal(size=(9, 9)))
        score = disco_classification(features, labels, n_groups=6, pca_components=0)
        rotated_score = disco_classification(
            features @ rotation, labels, n_groups=6, pca_components=0
        )
        self.assertTrue(np.isclose(score, rotated_score, atol=1e-9))

    def test_separable_features_score_above_shuffled_labels(self):
        rng = np.random.default_rng(2)
        labels = np.repeat(np.arange(3), 30)
        features = np.eye(3)[labels] * 5.0 + rng.normal(scale=0.2, size=(90, 3))
        shuffled = rng.permutation(labels)
        self.assertGreater(
            disco_classification(features, labels, n_groups=3, pca_components=0),
            disco_classification(features, shuffled, n_groups=3, pca_components=0),
        )


if __name__ == "__main__":
    unittest.main()
