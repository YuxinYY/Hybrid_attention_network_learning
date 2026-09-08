"""Common evaluation and prediction persistence for all model families."""
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)


def classification_metrics(truth, probabilities):
    truth = np.asarray(truth, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if truth.ndim != 1 or truth.size == 0 or truth.shape != probabilities.shape:
        raise ValueError('Expected nonempty, aligned labels and probabilities')
    if not np.isin(truth, [0, 1]).all() or not np.isfinite(probabilities).all():
        raise ValueError('Expected binary labels and finite probabilities')
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Probabilities must be within [0, 1]')
    predictions = (probabilities > 0.5).astype(int)
    report = classification_report(truth, predictions, labels=[0, 1],
                                    target_names=['Safe (0)', 'Risk (1)'],
                                    output_dict=True, zero_division=0)
    return dict(test_samples=len(truth), accuracy=accuracy_score(truth, predictions),
                balanced_accuracy=balanced_accuracy_score(truth, predictions),
                roc_auc=roc_auc_score(truth, probabilities) if len(set(truth)) == 2 else None,
                average_precision=average_precision_score(truth, probabilities) if sum(truth) else None,
                confusion_matrix=confusion_matrix(truth, predictions, labels=[0, 1]).tolist(),
                classification_report=report,
                always_negative_accuracy=float(1-np.mean(truth)),
                no_skill_average_precision=float(np.mean(truth)))


def predict_neural(model, loader, device):
    model.eval()
    truth, probabilities = [], []
    loss_sum, weight_sum = 0., 0.
    weights = torch.tensor([1., 1.4], device=device)
    with torch.no_grad():
        for batch in loader:
            y = batch['y'].to(device)
            logits, _, _ = model(batch['x_text'].to(device), batch['x_mask'].to(device))
            loss_sum += torch.nn.functional.cross_entropy(
                logits, y, weight=weights, reduction='sum').item()
            weight_sum += weights[y].sum().item()
            truth.extend(y.cpu().tolist())
            probabilities.extend(logits.softmax(dim=1)[:, 1].cpu().tolist())
    metrics = classification_metrics(truth, probabilities)
    metrics['weighted_cross_entropy'] = loss_sum / weight_sum
    return metrics, np.asarray(probabilities)


def save_evaluation(output, samples, probabilities, metrics, prefix='test'):
    if len(samples) != len(probabilities):
        raise ValueError('Predictions do not align with samples')
    (output / f'{prefix}_metrics.json').write_text(json.dumps(metrics, indent=2))
    pd.DataFrame([dict(ticker=s[0], date=s[1], label=int(s[3]),
                       prediction=int(q > 0.5), risk_probability=float(q))
                  for s, q in zip(samples, probabilities)]
                 ).to_csv(output / f'{prefix}_predictions.csv', index=False)
