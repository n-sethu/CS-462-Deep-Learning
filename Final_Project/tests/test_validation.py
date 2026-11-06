import unittest
import numpy as np
from src.metrics import risk_curve, metrics
from src.validation import output_features, probabilities, distances, FEATURES


class ValidationTests(unittest.TestCase):
    def test_hand_computed_and_ties(self):
        errors = np.array([1, 0, 1, 0])
        np.testing.assert_allclose(risk_curve(errors, [0, 0, 0, 0], [3, 0, 2, 1]), [0, 0, 1/3, 1/2])
        self.assertAlmostEqual(metrics(errors, [3, 0, 2, 1], [0, 1, 2, 3])['aurc'], (1/3 + 1/2)/4)

    def test_perfect_random_and_worst_ranker(self):
        errors = np.tile([0, 1], 500)
        ids = np.arange(len(errors))
        perfect = metrics(errors, errors, ids)['aurc']
        worst = metrics(errors, -errors, ids)['aurc']
        random = metrics(errors, np.random.default_rng(42).random(len(errors)), ids)['aurc']
        self.assertLess(perfect, random)
        self.assertLess(random, worst)
        self.assertAlmostEqual(random, .5, delta=.05)

    def test_softmax_average_and_no_labels(self):
        logits = np.array([[[20., 0.], [0., 2.], [0., 2.]]])
        features = output_features(logits)
        self.assertEqual(features['prediction'][0], 1)
        self.assertEqual(logits.mean(1).argmax(1)[0], 0)
        self.assertFalse(features['unanimous'][0])
        self.assertGreater(features['mutual_information'][0], 0)
        np.testing.assert_allclose(probabilities(logits).sum(-1), 1)
        for columns in FEATURES.values():
            self.assertFalse(set(columns) & {'label', 'error', 'id'})

    def test_predicted_class_own_model_distance(self):
        features = np.array([[[1., 2.], [4., 5.], [7., 8.]]])
        centroids = np.zeros((3, 2, 2))
        centroids[:, 1] = features[0]
        np.testing.assert_array_equal(distances(features, np.array([1]), centroids), [[0, 0, 0]])
        self.assertTrue((distances(features, np.array([0]), centroids) > 0).all())


if __name__ == '__main__':
    unittest.main()
