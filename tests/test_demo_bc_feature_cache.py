from pathlib import Path
import tempfile
import types
import unittest

import numpy as np

from tools.train_demo_bc_cached import read_feature_cache


class CacheIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runs = [types.SimpleNamespace(actions=np.array([0, 1, 2]))]
        self.ids = [(0, 1), (0, 2)]

    def tearDown(self):
        self.temp.cleanup()

    def write(self, *, features=None, labels=None, origins=None):
        np.savez(self.root/'cache.npz',
                 features=np.zeros((2, 192), dtype=np.float32) if features is None else features,
                 labels=np.array([1, 2]) if labels is None else labels,
                 origins=np.array(self.ids) if origins is None else origins)

    def test_same_samples_and_labels_can_be_reused(self):
        self.write()
        x, y, origins = read_feature_cache(self.root, 'cache.npz', self.runs, self.ids)
        self.assertEqual(x.shape, (2, 192))
        np.testing.assert_array_equal(y, [1, 2])

    def test_wrong_labels_are_rejected(self):
        self.write(labels=np.array([2, 1]))
        with self.assertRaisesRegex(ValueError, 'labels changed'):
            read_feature_cache(self.root, 'cache.npz', self.runs, self.ids)

    def test_wrong_sample_selection_is_rejected(self):
        self.write(origins=np.array([(0, 0), (0, 2)]))
        with self.assertRaisesRegex(ValueError, 'selection changed'):
            read_feature_cache(self.root, 'cache.npz', self.runs, self.ids)

    def test_nonfinite_features_are_rejected(self):
        x = np.zeros((2, 192), dtype=np.float32)
        x[0, 0] = float('nan')
        self.write(features=x)
        with self.assertRaisesRegex(ValueError, 'dimensions or values'):
            read_feature_cache(self.root, 'cache.npz', self.runs, self.ids)

    def test_wrong_extractor_width_is_rejected(self):
        self.write(features=np.zeros((2, 64)))
        with self.assertRaisesRegex(ValueError, 'dimensions or values'):
            read_feature_cache(self.root, 'cache.npz', self.runs, self.ids)
