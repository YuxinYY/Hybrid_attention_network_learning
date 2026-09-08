"""Train/evaluate HAN with daily top-20% labels and save a reproducible experiment."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from data_processing.scripts.labels import (
    LABEL_SCHEME, compute_future_realized_vol, daily_top_fraction_labels, relabel_samples,
)
from dataset_utils import HandlersDataset
from model import HAN_Classification
from train import ClassificationTrainer
from evaluation import predict_neural, save_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data_processing'))
    parser.add_argument('--stocks', type=Path, default=Path('stocks.csv'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--compare-baselines', action='store_true',
                        help='Also train/evaluate CNN, logistic regression, and temporal Transformer')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    output = args.output_dir or Path('experiments') / (
        'daily_top20_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    )
    output.mkdir(parents=True, exist_ok=False)
    config = torch.load(args.data_dir / 'config.pt', weights_only=False)
    day_dict = torch.load(args.data_dir / 'day_dict.pt', weights_only=False)
    labels = daily_top_fraction_labels(compute_future_realized_vol(pd.read_csv(args.stocks)))
    labels.to_csv(output / 'daily_labels.csv', index=False)
    splits, counts, sources = {}, {}, {}
    boundaries = {'train': config.get('val_start', '2023-08-01'),
                  'val': config.get('test_start', '2023-11-01'), 'test': None}
    # Always regenerate targets from returns; old median labels are never reused.
    for name, boundary in boundaries.items():
        source = args.data_dir / f'{name}_samples.pt'
        original = torch.load(source, weights_only=False)
        samples = relabel_samples(original, labels, boundary)
        if not samples:
            raise ValueError(f'No usable {name} samples after relabeling')
        splits[name] = samples
        torch.save(samples, output / f'{name}_samples.pt')
        positives = sum(s[3] for s in samples)
        counts[name] = dict(samples=len(samples), positive=positives,
                            negative=len(samples)-positives,
                            positive_rate=positives/len(samples),
                            dropped=len(original)-len(samples),
                            first_date=str(min(s[1] for s in samples)),
                            last_date=str(max(s[1] for s in samples)))
        sources[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        print(name, counts[name], flush=True)
    for source in [args.stocks, args.data_dir / 'day_dict.pt', args.data_dir / 'config.pt',
                   Path(__file__), Path('train.py'), Path('model.py'), Path('dataset_utils.py'),
                   Path('data_processing/scripts/labels.py'), Path('evaluation.py'),
                   Path('baseline_models.py'), Path('baseline_comparison.py')]:
        sources[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    datasets = {name: HandlersDataset(s, day_dict, L=50, E=config['E'], D=0)
                for name, s in splits.items()}
    targets = np.array([s[3] for s in splits['train']])
    class_counts = np.bincount(targets, minlength=2)
    if (class_counts == 0).any():
        raise ValueError('Training split must contain both classes')
    sampler = WeightedRandomSampler(torch.tensor(1.0/class_counts[targets]),
                                    len(targets), replacement=True)
    loaders = {name: DataLoader(ds, batch_size=64, num_workers=0,
                                pin_memory=device.type == 'cuda',
                                sampler=sampler if name == 'train' else None,
                                drop_last=name == 'train' and len(ds) >= 64)
               for name, ds in datasets.items()}
    model = HAN_Classification(embedding_dim=config['E'], gru_hidden_dim=64,
                               gru_num_layers=1, prediction_hidden_dim=32,
                               num_classes=2, dropout=0.4).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 1.4], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)
    metadata = dict(label_scheme=LABEL_SCHEME, positive_fraction=0.2,
                    ranking_universe='All stocks with complete forward returns in stocks CSV',
                    universe_tickers=sorted(labels.ticker.unique().tolist()),
                    tie_break='ticker ascending', positive_count='ceil(0.2 * daily N)',
                    seed=args.seed, device=str(device), epochs_requested=args.epochs,
                    learning_rate=5e-5, batch_size=64, weight_decay=1e-4,
                    dropout=0.4, class_weights=[1., 1.4], sampler='inverse frequency',
                    split_counts=counts, purge_boundaries=boundaries, source_sha256=sources,
                    text_source='Reused saved day_dict and sample lookback windows',
                    decision_rule='argmax logits (positive probability > 0.5)',
                    checkpoint_selection='maximum validation Risk F1',
                    compare_baselines=args.compare_baselines)
    (output / 'config.json').write_text(json.dumps(metadata, indent=2))
    trainer = ClassificationTrainer(model, loaders['train'], loaders['val'], optimizer,
                                    scheduler, criterion, device, output)
    trainer.train(num_epochs=args.epochs, patience=5)
    (output / 'history.json').write_text(json.dumps(trainer.history, indent=2))
    model.load_state_dict(torch.load(output / 'best_model.pt', map_location=device, weights_only=True))
    metrics, probabilities = predict_neural(model, loaders['test'], device)
    metrics.update(best_epoch=trainer.best_epoch, epochs_completed=len(trainer.history['train_loss']))
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    save_evaluation(output, splits['test'], probabilities, metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    if args.compare_baselines:
        from baseline_comparison import compare_baselines
        compare_baselines(model, metrics, probabilities, datasets, splits, output,
                          config['E'], args.seed, args.epochs, device)
    print(f'Experiment saved to {output}', flush=True)


if __name__ == '__main__':
    main()
