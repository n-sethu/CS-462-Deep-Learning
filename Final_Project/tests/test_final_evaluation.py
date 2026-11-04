import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.final_evaluation import paired_bootstrap, feature_table, FrozenDetectors, freeze
from src.metrics import risk_curve


class FinalTests(unittest.TestCase):
    def test_identical_detectors_have_zero_paired_interval(self):
        result = paired_bootstrap([0, 1, 0, 1], [.1, .4, .2, .3], [.1, .4, .2, .3], [0, 1, 2, 3], resamples=100)
        self.assertEqual(result['delta_aurc'], 0)
        self.assertEqual(result['ci_low'], 0)
        self.assertEqual(result['ci_high'], 0)
        self.assertFalse(result['supports_improvement'])

    def test_cluster_bootstrap_resamples_source_copies_together(self):
        e = np.array([0, 1, 0, 1, 0, 1])
        a, b = np.array([0, 1, 2, 3, 4, 5]), np.array([5, 4, 3, 2, 1, 0])
        groups = np.array([0, 1, 0, 1, 0, 1])
        result = paired_bootstrap(e, a, b, np.arange(6), resamples=30, seed=9, groups=groups)
        rng = np.random.default_rng(9)
        expected = []
        for _ in range(30):
            draw = rng.integers(0, 2, 2)
            ix = np.concatenate([np.flatnonzero(groups == g) for g in draw])
            expected.append(risk_curve(e[ix], a[ix], ix).mean()-risk_curve(e[ix], b[ix], ix).mean())
        np.testing.assert_allclose([result['ci_low'], result['ci_high']], np.quantile(expected, [.025, .975]))

    def test_features_ignore_labels(self):
        cache = dict(id=np.arange(3), logits=np.arange(90).reshape(3, 3, 10), features=np.ones((3, 3, 512)), label=np.zeros(3))
        stats = dict(centroids=np.zeros((3, 10, 512)), mean=np.zeros(3), std=np.ones(3))
        first = feature_table(cache, stats)
        cache['label'] = np.ones(3)
        pd.testing.assert_frame_equal(first, feature_table(cache, stats))
        self.assertNotIn('label', first.columns)

    def test_frozen_reload_reproduces_validation_scores(self):
        import json
        root = Path('results/validation')
        if not (root / 'frozen_protocol.json').exists():
            self.skipTest('Saved validation artifacts not present')
        detectors = FrozenDetectors(root, json.loads((root / 'frozen_protocol.json').read_text()))
        features = pd.read_csv(root / 'val_features.csv')
        saved = pd.read_csv(root / 'validation_scores.csv')
        for name, score in detectors.predict(features).items():
            np.testing.assert_allclose(score, saved[name], atol=2e-7, rtol=2e-6, err_msg=name)


if __name__ == '__main__':
    unittest.main()

class FreezeIntegrityTests(unittest.TestCase):
    def test_artifact_tampering_is_rejected(self):
        import json
        from src.validation import sha
        from src.final_evaluation import verify_artifacts
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'model.pt'
            model.write_bytes(b'frozen')
            checkpoint = root / 'cnn.pt'
            checkpoint.write_bytes(b'cnn')
            (root / 'frozen_protocol.json').write_text(json.dumps({'files': {'model.pt': sha(model)}}))
            (root / 'cache_manifest.json').write_text(json.dumps({'checkpoints': {str(checkpoint): sha(checkpoint)}}))
            verify_artifacts(root)
            model.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Frozen artifact changed'):
                verify_artifacts(root)

    def test_validation_threshold_changes_cannot_replace_seal(self):
        with tempfile.TemporaryDirectory() as directory:
            validation, output = Path(directory) / 'validation', Path(directory) / 'final'
            validation.mkdir()
            output.mkdir()
            (validation / 'frozen_protocol.json').write_text('{}')
            (validation / 'cache_manifest.json').write_text('{}')
            scores = validation / 'validation_scores.csv'
            scores.write_text('id,error,A_lr\n0,0,0.1\n1,1,0.9\n')
            protocol = {'selected_mlp_seeds': dict(A=33, B=33, C=33, D=22)}
            with patch('src.final_evaluation.verify_artifacts', return_value=(protocol, {})):
                freeze(validation, output, 2000)
                freeze(validation, output, 2000)
                scores.write_text('id,error,A_lr\n0,0,0.2\n1,1,0.8\n')
                with self.assertRaisesRegex(ValueError, 'Final protocol changed'):
                    freeze(validation, output, 2000)
