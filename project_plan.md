# Current Project Plan

This document describes the active daily-top-20% experiment and the latest
completed comparison of HAN, logistic regression, CNN, and Transformer.
It is a current implementation reference, not a history of earlier experiments.

## Project and File Locations

- Local code: `/Users/yuxin/Documents/Local_InterviewProjects/Hybrid_attention_network_learning`
- External project/data: `/Volumes/T9/InterviewProjects/Hybrid_attention_network_learning`
- Latest completed run: `experiments/daily_top20_baselines_seed42/` under the
  external project directory.

Code, configuration, documentation, and tests have been synchronized locally.
Data, cached embeddings, model weights, and experiment artifacts remain on the
external drive; they were not copied during that synchronization. The runner
can read external data through explicit command-line paths.

## Data Pipeline and Label Definition

The task is to predict which stocks will be among the most volatile on a
given date using the preceding Reddit discussion. The active label scheme is
`daily_top20_forward_rv5_v1`, implemented in
[labels.py](data_processing/scripts/labels.py).

For stock `i` at anchor date `t`, calculate:

```text
RV_5(i, t) = sqrt(RET(i, t+1)^2 + ... + RET(i, t+5)^2)
```

The five returns are the next five observations within that stock, excluding
the anchor return. Missing or incomplete forward windows are excluded.

On each date, rank the distinct stocks with valid forward volatility in
`stocks.csv`. The highest `ceil(0.2 * N)` stocks receive `Risk = 1`; the
others receive `Safe = 0`. Equal volatility is resolved by ascending ticker.
The current universe has 20 stocks, so four are positive per date. Ranking
happens before joining to Reddit posts, so posting frequency does not affect
the labels. The subset with usable text windows may have a different positive
percentage.

[pipeline.py](data_processing/scripts/pipeline.py) performs the following:

1. Read Reddit data from `--submissionandcomments_dir` and market data from
   `--stocks`, configured in [config.py](config.py). Reddit fields include
   `id, title, selftext, body, date, score`; stock fields include
   `date, ticker, name, RET, VOL`.
2. Match posts to stocks with FlashText ticker matching, GLiNER company
   extraction, company-name mapping, and an ambiguity blacklist.
3. Join matched text to stock-date labels and calculate finance-keyword,
   word, and numeric-token counts.
4. Encode text using the local pretrained FinBERT. Long text is divided into
   segments of up to 510 content tokens plus special tokens; segment CLS
   embeddings are averaged. Parquet chunks are stored under
   `data_processing/embedding_output_daily_top20/`.
5. Sort each stock-day's posts by finance-keyword count, word count, and
   Reddit score; retain up to 50 embeddings.
6. Build samples with the preceding 20 business weekdays, excluding the
   anchor date. The weekday calendar is not an exchange holiday calendar.
7. Split by anchor date and remove samples whose label horizons cross into
   the next split.

Outputs in `data_processing/` are `day_dict.pt`,
`train_samples.pt`, `val_samples.pt`, `test_samples.pt`, and `config.pt`.
Supporting ingestion scripts are `streaming_filter.py`,
`prepare_reddit_csv.py`, and `fetch_stock_data.py`.

## Shared Experimental Data

[run.py](run.py) reads the cached embeddings and sample windows, recalculates
labels from stock returns, and writes the relabeled splits to a new run
directory. The latest experiment reused existing text embeddings and windows;
it did not re-encode the full Reddit corpus.

All four models receive the same stock-date samples, labels, and frozen
768-dimensional FinBERT features. FinBERT is not fine-tuned.
[HandlersDataset](dataset_utils.py) pads to 50 posts per day and provides a
Boolean mask. The learned models use text only (`D = 0`); stored numerical
day features are not inputs.

| Split | Anchor-date rule | Retained samples | Positive | Negative | Removed at boundary |
| --- | --- | ---: | ---: | ---: | ---: |
| Train | before 2023-08-01 | 702 | 129 | 573 | 41 |
| Validation | 2023-08-01 through 2023-10-31 | 281 | 57 | 224 | 29 |
| Test | on or after 2023-11-01 | 223 | 41 | 182 | 0 |

The latest test set covers 2023-11-01 through 2023-12-29. A training label's
forward horizon must end before validation begins; a validation label's
horizon must end before test begins. Since a daily rank depends on all stocks
on that date, this check uses the latest forward-window end date in the daily
ranking universe.

## Models and Training

| Model | Feature aggregation and classifier | Trainable parameters |
| --- | --- | ---: |
| HAN | Learned post attention → one-layer BiGRU, hidden size 64 per direction → temporal attention → 128/32/32/2 classification head | 440,706 |
| Logistic regression | Masked mean of posts within each day → equal mean of observed days → StandardScaler → L2 logistic regression | 769 |
| CNN | Masked daily mean → 3-day and 5-day Conv1d, 64 channels each → temporal max pooling → classification head | 397,538 |
| Transformer | Masked daily mean → projection to width 128 → learned positions and CLS token → two encoder layers, four heads, feedforward width 256 → classification head | 370,402 |

HAN is defined in [model.py](model.py). CNN and Transformer are defined in
[baseline_models.py](baseline_models.py). The Transformer uses
[PyTorch TransformerEncoder](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html).
Its missing days keep their positions but are masked from attention keys;
CLS remains valid for an entirely empty window. All neural classifiers are
initialized from scratch; parameter counts exclude frozen FinBERT.

### Neural training and checkpoint selection

HAN, CNN, and Transformer share [ClassificationTrainer](train.py):

- Seed 42 for the completed comparison; CPU execution.
- At most 10 epochs, batch size 64, dropout 0.4.
- AdamW: learning rate `5e-5`, weight decay `1e-4`.
- Inverse-class-frequency sampling with replacement for training only.
- Cross-entropy weights `[1.0, 1.4]`; gradient norm clipping at 5.
- ReduceLROnPlateau on validation loss: factor 0.5, patience 2,
  minimum learning rate `1e-6`.
- Early stopping after five consecutive epochs without validation-loss
  improvement.
- Save the checkpoint with maximum validation positive-class F1. The first
  epoch is saved even if F1 is zero; ties retain the earlier checkpoint.

Validation runs after each training epoch. Early stopping and learning-rate
scheduling use validation loss; checkpoint selection uses validation F1.
Training batches drop the final incomplete batch when the dataset has at least
64 samples. Validation and test retain every sample and do not resample.

### Logistic regression fitting and selection

[baseline_comparison.py](baseline_comparison.py) fits a StandardScaler and
logistic regression on training data only. It tries `C = [0.01, 0.1, 1, 10]`
with `lbfgs` and at most 2,000 iterations, then selects the converged candidate
with the highest validation positive F1. Ties favor the smaller C.

Its class weights are `N / (2*n_negative)` and
`1.4*N / (2*n_positive)`, matching the expected relative class emphasis of
the neural sampling/loss combination. No training/validation merger or test
fitting occurs after C selection.

| Model | Selected checkpoint / setting | Epochs completed | Validation positive F1 |
| --- | --- | ---: | ---: |
| HAN | epoch 6 | 6 | 0.3946 |
| Logistic regression | C = 0.01 | not applicable | 0.4853 |
| CNN | epoch 5 | 10 | 0.3669 |
| Transformer | epoch 2 | 6 | 0.3383 |

## Evaluation Procedure

The shared implementation is [evaluation.py](evaluation.py), invoked by
[run.py](run.py) and [baseline_comparison.py](baseline_comparison.py).

1. **Freeze the selected model.** Reload the validation-selected checkpoint
   for HAN/CNN/Transformer, or use the validation-selected fitted LR pipeline.
2. **Generate probabilities on the ordered test split.** Neural models run in
   `model.eval()` mode under `torch.no_grad()`, with dropout disabled.
   Positive probability is `softmax(logits)[:, 1]`. LR uses
   `predict_proba(X)[:, 1]` after the training-fitted scaler.
3. **Apply the common classification threshold.** Every model uses
   `p(Risk) > 0.5 → 1` and `p(Risk) <= 0.5 → 0`. There is no
   model-specific threshold optimization and no daily top-k selection of
   predictions. The probability threshold is separate from the daily top-20%
   rule used to create ground-truth labels.
4. **Calculate metrics over all test stock-date samples together.** These
   are pooled metrics, not averages of per-day or per-stock metrics. Every
   model is evaluated on the same 223 samples in the same order.
5. **Save the predictions and metrics.** Each prediction row records ticker,
   anchor date, true label, predicted label, and positive probability.

Test data is not used to select checkpoints, C, scalers, or decision
thresholds. The comparison is a fixed-configuration, single-seed experiment,
not an exhaustive architecture/hyperparameter search.

### Metric definitions

| Metric | Calculation and interpretation |
| --- | --- |
| Accuracy | `(TP + TN) / N`; fraction of all correct predictions |
| Positive precision | `TP / (TP + FP)`; fraction of predicted positives that are correct |
| Positive recall | `TP / (TP + FN)`; fraction of actual positives detected |
| Positive F1 | Harmonic mean of positive precision and recall |
| Balanced accuracy | Mean of positive recall and negative recall |
| ROC-AUC | Computed from continuous positive probabilities, independent of the fixed 0.5 threshold |
| Average precision (AP) | Precision-recall summary computed from continuous probabilities via scikit-learn average_precision_score |
| Confusion matrix | Rows actual Safe/Risk, columns predicted Safe/Risk: `[[TN, FP], [FN, TP]]` |

The classification report also includes each class's metrics and macro/weighted
averages; undefined precision/F1 values are set to zero. Neural evaluation
additionally saves weighted cross-entropy: summed per-sample weighted losses
divided by the sum of target-class weights over the test set.

For this test set, always predicting negative yields 81.61% accuracy but zero
positive recall/F1. Positive prevalence is 18.39%, the no-skill AP reference.
Thus accuracy alone is insufficient to judge detection of positive cases.

## Latest Evaluation Results

Source: the saved
[comparison report](/Volumes/T9/InterviewProjects/Hybrid_attention_network_learning/experiments/daily_top20_baselines_seed42/comparison.md)
and its per-model prediction/metric files.

| Model | Accuracy | Positive precision | Positive recall | Positive F1 | ROC-AUC | AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HAN | 55.61% | 22.12% | 56.10% | 0.3172 | 0.6388 | 0.3186 |
| Logistic regression | 76.68% | 38.78% | 46.34% | 0.4222 | 0.6316 | 0.2787 |
| CNN | 35.43% | 19.16% | 78.05% | 0.3077 | 0.5937 | 0.2698 |
| Transformer | 18.39% | 18.10% | 97.56% | 0.3053 | 0.5125 | 0.1860 |

| Model | TN | FP | FN | TP | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| HAN | 101 | 81 | 18 | 23 | 55.80% |
| Logistic regression | 152 | 30 | 22 | 19 | 64.93% |
| CNN | 47 | 135 | 9 | 32 | 51.94% |
| Transformer | 1 | 181 | 1 | 40 | 49.06% |

LR has the best positive F1 and precision at the default threshold. HAN has
the highest ROC-AUC and AP in this run, but does not beat LR on thresholded
classification. CNN and Transformer predict many positives; the Transformer
flags 221 of 223 samples. Sampling and loss weights emphasize positives, but
their causal contribution has not been isolated.

This small, single-seed text subset does not establish general architectural
superiority. HAN's ROC-AUC advantage over LR (0.6388 versus 0.6316) is small.
HAN also learns post aggregation whereas the baselines use means, so the
comparison does not isolate the temporal layer alone.

## Reproduction and Saved Artifacts

Run from the local project root with a compatible Python environment, reading
the existing external-drive tensors and stock data:

```bash
python run.py \
  --data-dir /Volumes/T9/InterviewProjects/Hybrid_attention_network_learning/data_processing \
  --stocks /Volumes/T9/InterviewProjects/Hybrid_attention_network_learning/stocks.csv \
  --seed 42 --compare-baselines
```

The default output is a new local `experiments/daily_top20_<timestamp>/`
directory. `--output-dir <new-directory>` chooses another location;
`--epochs` defaults to 10. Omitting `--compare-baselines` runs HAN alone.
These command-line paths do not modify the underlying data configuration.

The completed comparison artifacts currently remain in the external run
directory identified above:

- Run root: settings/input/code hashes in `config.json`, daily labels and
  relabeled split tensors; HAN checkpoint, history, validation/test predictions,
  `metrics.json`, `test_metrics.json`, and `validation_metrics.json`.
- `baselines/logistic_regression/`: `model.joblib` (including the scaler),
  C candidates/selection, validation/test metrics, and predictions.
- `baselines/cnn/` and `baselines/transformer/`: `best_model.pt`,
  `config.json`, `history.json`, `training.log`, validation/test metrics,
  and predictions.
- Run root summaries: `comparison.csv`, `comparison.json`, and
  `comparison.md`.

Nine tests passed for labels, pooling/masking, empty windows, metric
thresholding, and LR train-only scaling/test-label independence. Saved neural
baseline checkpoints reproduced their predictions, and saved comparison
metrics were checked against the aligned prediction files.

## Next Experiments

1. Compare thresholds chosen on validation data with the fixed 0.5 rule.
2. Assess sensitivity to random seeds and sampling/class weights.
3. Add market-data baselines and controlled post/temporal aggregation ablations.
4. Expand the text sample and inspect attention weights and false positives.
