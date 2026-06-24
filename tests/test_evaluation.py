import unittest

import numpy as np

from firecls.evaluation import classification_metrics, prediction_agreement


class EvaluationTests(unittest.TestCase):
    def test_metrics_preserve_zero_support_class(self):
        metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], ["a", "b", "missing"])
        self.assertEqual(metrics["samples"], 4)
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["per_class"]["missing"]["support"], 0)
        self.assertEqual(len(metrics["confusion_matrix"]), 3)

    def test_agreement_counts_changed_predictions(self):
        reference = np.array([[3.0, 1.0], [0.1, 0.9]], dtype=np.float32)
        candidate = np.array([[2.9, 1.1], [1.0, 0.8]], dtype=np.float32)
        agreement = prediction_agreement(reference, candidate)
        self.assertEqual(agreement["changed_predictions"], 1)
        self.assertAlmostEqual(agreement["top1_agreement"], 0.5)


if __name__ == "__main__":
    unittest.main()
