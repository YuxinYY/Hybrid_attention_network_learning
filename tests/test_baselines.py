import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import torch

from baseline_models import TemporalCNN, TemporalTransformer, daily_mean, window_mean
from baseline_comparison import train_logistic
from evaluation import classification_metrics


class BaselineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(2)

    def test_pooling_ignores_padding_and_empty_days(self):
        x = torch.tensor([[[[2.], [4.], [999.]], [[8.], [999.], [999.]],
                           [[999.], [999.], [999.]]]])
        mask = torch.tensor([[[True, True, False], [True, False, False], [False]*3]])
        means, _ = daily_mean(x, mask)
        torch.testing.assert_close(means.flatten(), torch.tensor([3., 8., 0.]))
        torch.testing.assert_close(window_mean(x, mask), torch.tensor([[5.5]]))

    def test_neural_baselines_ignore_padded_values_and_handle_empty_windows(self):
        x = torch.randn(3, 20, 4, 8)
        mask = torch.rand(3, 20, 4) > 0.5
        mask[0] = False
        altered = x.masked_fill(~mask.unsqueeze(-1), 1e8)
        for cls in (TemporalCNN, TemporalTransformer):
            model = cls(embedding_dim=8, dropout=0).eval()
            actual = model(x, mask)[0]
            self.assertEqual(actual.shape, (3, 2))
            self.assertTrue(torch.isfinite(actual).all())
            torch.testing.assert_close(actual, model(altered, mask)[0])
            actual.sum().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                                if p.grad is not None))

    def test_metric_threshold_and_zero_positive_predictions(self):
        metrics = classification_metrics([0, 0, 1, 1], [0.1, 0.5, 0.2, 0.4])
        self.assertEqual(metrics['confusion_matrix'], [[2, 0], [2, 0]])
        self.assertEqual(metrics['classification_report']['Risk (1)']['f1-score'], 0)
        with self.assertRaises(ValueError):
            classification_metrics([0, 1], [0.2, float('nan')])

    def test_logistic_scaling_and_selection_do_not_use_test_labels(self):
        rng = np.random.default_rng(9)
        train_x = rng.normal(size=(60, 4))
        train_y = (train_x[:, 0] > 0).astype(int)
        val_x = rng.normal(size=(20, 4))
        test_x = rng.normal(size=(20, 4)) + 100
        features = {'train': (train_x, train_y), 'val': (val_x, (val_x[:, 0] > 0).astype(int)),
                    'test': (test_x, np.array([0, 1]*10))}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            first = path / 'first'
            first.mkdir()
            model, _, _, probs = train_logistic(features, first, 42)
            np.testing.assert_allclose(model[0].mean_, train_x.mean(axis=0))
            restored = joblib.load(first / 'model.joblib')
            np.testing.assert_allclose(restored.predict_proba(test_x)[:, 1], probs)
            features['test'] = (test_x, 1-features['test'][1])
            second = path / 'second'
            second.mkdir()
            changed, _, _, changed_probs = train_logistic(features, second, 42)
            self.assertEqual(model[-1].C, changed[-1].C)
            np.testing.assert_allclose(probs, changed_probs)


if __name__ == '__main__':
    unittest.main()
