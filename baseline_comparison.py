"""Train baselines on HAN splits; select on validation and report untouched test scores."""
import contextlib
import json
import random
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, WeightedRandomSampler

from baseline_models import TemporalCNN, TemporalTransformer, window_mean
from evaluation import classification_metrics, predict_neural, save_evaluation
from train import ClassificationTrainer


def pooled_features(loader):
    features, labels = [], []
    for batch in loader:
        features.append(window_mean(batch['x_text'], batch['x_mask']).numpy())
        labels.extend(batch['y'].tolist())
    return np.concatenate(features), np.asarray(labels)


def train_logistic(features, output, seed):
    x_train, y_train = features['train']
    x_val, y_val = features['val']
    counts = np.bincount(y_train, minlength=2)
    # Match expected class emphasis of inverse-frequency sampling + [1, 1.4] CE.
    weights = {0: len(y_train)/(2*int(counts[0])),
               1: 1.4*len(y_train)/(2*int(counts[1]))}
    candidates, best, best_score = [], None, -1.
    for c in (0.01, 0.1, 1.0, 10.0):
        estimator = make_pipeline(StandardScaler(), LogisticRegression(
            C=c, solver='lbfgs', max_iter=2000, class_weight=weights, random_state=seed))
        # Scaling is fitted exclusively on training data for every candidate.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            estimator.fit(x_train, y_train)
        val = classification_metrics(y_val, estimator.predict_proba(x_val)[:, 1])
        score = val['classification_report']['Risk (1)']['f1-score']
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        candidates.append(dict(C=c, validation_f1=score, converged=converged))
        if converged and score > best_score:
            best, best_score, best_c = estimator, score, c
    if best is None:
        raise RuntimeError('No logistic regression candidate converged')
    joblib.dump(best, output / 'model.joblib')
    settings = dict(model='logistic_regression', seed=seed, C=best_c,
                    class_weight=weights, candidates=candidates,
                    features='Mean of observed daily mean FinBERT vectors; train-only StandardScaler',
                    selection='Maximum validation Risk F1 at threshold 0.5; C ties favor smaller C')
    (output / 'config.json').write_text(json.dumps(settings, indent=2))
    val_metrics = classification_metrics(y_val, best.predict_proba(x_val)[:, 1])
    x_test, y_test = features['test']
    p_test = best.predict_proba(x_test)[:, 1]
    return best, val_metrics, classification_metrics(y_test, p_test), p_test


def compare_baselines(han, han_test_metrics, han_test_probabilities,
                      datasets, splits, output, embedding_dim, seed, epochs, device):
    root = output / 'baselines'
    root.mkdir()
    eval_loaders = {name: DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
                    for name, ds in datasets.items()}
    han_val, han_val_p = predict_neural(han, eval_loaders['val'], device)
    save_evaluation(output, splits['val'], han_val_p, han_val, prefix='validation')
    rows = []

    def add_row(name, val, test, params, best_epoch=None):
        risk = test['classification_report']['Risk (1)']
        rows.append(dict(model=name, parameters=params, best_epoch=best_epoch,
                         validation_f1=val['classification_report']['Risk (1)']['f1-score'],
                         accuracy=test['accuracy'], balanced_accuracy=test['balanced_accuracy'],
                         precision=risk['precision'], recall=risk['recall'], f1=risk['f1-score'],
                         roc_auc=test['roc_auc'], average_precision=test['average_precision'],
                         test_samples=test['test_samples']))

    # Recalculate HAN through the shared scorer to guarantee the same threshold policy.
    han_test = classification_metrics([s[3] for s in splits['test']], han_test_probabilities)
    add_row('HAN', han_val, han_test, sum(p.numel() for p in han.parameters()),
            han_test_metrics['best_epoch'])
    features = {name: pooled_features(loader) for name, loader in eval_loaders.items()}
    destination = root / 'logistic_regression'
    destination.mkdir()
    print('Training logistic regression (validation selection over four C values)...', flush=True)
    lr, val, test, probs = train_logistic(features, destination, seed)
    save_evaluation(destination, splits['test'], probs, test)
    save_evaluation(destination, splits['val'], lr.predict_proba(features['val'][0])[:, 1],
                    val, prefix='validation')
    add_row('Logistic regression', val, test, int(lr[-1].coef_.size + lr[-1].intercept_.size))

    y_train = np.array([s[3] for s in splits['train']])
    counts = np.bincount(y_train, minlength=2)
    for name, model_class in [('CNN', TemporalCNN), ('Transformer', TemporalTransformer)]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        destination = root / name.lower()
        destination.mkdir()
        model = model_class(embedding_dim=embedding_dim, dropout=0.4).to(device)
        sampler = WeightedRandomSampler(torch.tensor(1.0/counts[y_train]),
                                        len(y_train), replacement=True)
        train_loader = DataLoader(datasets['train'], batch_size=64, sampler=sampler,
                                   num_workers=0, drop_last=len(y_train) >= 64)
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 1.4], device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)
        trainer = ClassificationTrainer(model, train_loader, eval_loaders['val'],
                                        optimizer, scheduler, criterion, device, destination)
        settings = dict(model=name, architecture=str(model), seed=seed, device=str(device),
                        learning_rate=5e-5, weight_decay=1e-4, batch_size=64,
                        dropout=0.4, class_weights=[1., 1.4], sampler='inverse frequency',
                        max_epochs=epochs, early_stopping='validation loss; patience=5',
                        selection='Maximum validation Risk F1; ties keep first epoch',
                        decision_threshold=0.5)
        (destination / 'config.json').write_text(json.dumps(settings, indent=2))
        print(f'Training {name}; detailed log: {destination / "training.log"}', flush=True)
        with (destination / 'training.log').open('w') as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                trainer.train(num_epochs=epochs, patience=5)
        (destination / 'history.json').write_text(json.dumps(trainer.history, indent=2))
        model.load_state_dict(torch.load(destination / 'best_model.pt', map_location=device,
                                         weights_only=True))
        val, val_p = predict_neural(model, eval_loaders['val'], device)
        test, probs = predict_neural(model, eval_loaders['test'], device)
        test.update(best_epoch=trainer.best_epoch,
                    epochs_completed=len(trainer.history['train_loss']))
        save_evaluation(destination, splits['val'], val_p, val, prefix='validation')
        save_evaluation(destination, splits['test'], probs, test)
        add_row(name, val, test, sum(p.numel() for p in model.parameters()), trainer.best_epoch)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / 'comparison.csv', index=False)
    (output / 'comparison.json').write_text(json.dumps(rows, indent=2))
    text = '# HAN versus text baselines\n\n'
    text += ('All models share frozen FinBERT inputs, daily top-20% labels, purged temporal '
             'splits, and test probability threshold 0.5. Checkpoints and logistic C are '
             'selected on validation Risk F1 only. This is a single-seed comparison.\n\n')
    text += '| Model | Accuracy | Precision | Recall | Risk F1 | ROC-AUC | AP |\n'
    text += '| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n'
    for row in rows:
        text += (f'| {row["model"]} | {row["accuracy"]:.4f} | {row["precision"]:.4f} | '
                 f'{row["recall"]:.4f} | {row["f1"]:.4f} | {row["roc_auc"]:.4f} | '
                 f'{row["average_precision"]:.4f} |\n')
    text += ('\nCNN: daily means, 3/5-day convolutions. Logistic regression: mean of observed '
             'daily means, train-only scaling. Transformer: daily means, learned positions '
             'and CLS token, two layers, four heads, width 128. HAN uses learned post '
             'attention. Thus this compares complete aggregation/classification methods, '
             'not a controlled ablation of the temporal layer alone.\n')
    (output / 'comparison.md').write_text(text)
    print(comparison.to_string(index=False), flush=True)
    return comparison
