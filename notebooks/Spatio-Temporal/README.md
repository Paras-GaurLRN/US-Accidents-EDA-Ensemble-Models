# US Accidents — Weekly Forecasting Notebook Suite

This bundle contains **11 Jupyter notebooks** for the revised weekly accident-count forecasting objective. It replaces the previous event-level `Severity` classification workflow.

## Notebook execution order

1. `1 US Accidents - Weekly Model-Free PreProcessing.ipynb`
2. `2 US Accidents - Weekly Forecasting EDA and Split.ipynb`
3. `3I US Accidents - Incremental Weekly Preprocessing.ipynb`
4. `3N US Accidents - DCRNN Weekly Graph Preprocessing.ipynb`
5. `4IA US Accidents - SGDRegressor Squared Error.ipynb`
6. `4IB US Accidents - SGDRegressor Huber.ipynb`
7. `4IC US Accidents - SGDRegressor Epsilon Insensitive.ipynb`
8. `4ID US Accidents - MLPRegressor.ipynb`
9. `4N US Accidents - Weekly DCRNN.ipynb`
10. `5I US Accidents - Incremental Regression Ensemble.ipynb`
11. `6 US Accidents - Final Weekly Forecast Benchmarking.ipynb`

This is exactly 11 notebooks.

## Target definition

- One row per Monday-start local-calendar week, using the source `Start_Time` wall-clock value.
- `week_id` is a consecutive integer group identifier starting at 1.
- `week_start` is retained for calendar features and readable plots.
- The national target is the sum of counts across every observed state category. Missing state is kept as `__UNKNOWN__`; it is not silently dropped. Notebook 1 stops if any source row has a missing/unparseable `Start_Time`, because such a row cannot be assigned to a week without inventing a timestamp.
- The horizon is one weekly interval ahead.
- No `Severity` classification target is used.

The original dataset notes that `Start_Time` is local time and that data collection may have gaps. The notebooks therefore describe the target as **reported accident records**, not the complete number of real-world accidents. The exact Monday-local-week convention is written into preprocessing metadata.

## Geographic representation

- Stream I predicts the national weekly total directly.
- Stream N predicts state-level weekly totals on a state-adjacency graph. The state forecasts are summed to produce the same national target.
- The old hourly California / top-128 grid filter is removed. There is no node cap and no accident-level severity filter.
- State labels present in the source are all included; any unrecognized label remains as a self-loop-only node and is listed in graph metadata.

## Train/test and validation policy

- Notebook 2 creates a chronological 80:20 split by ordered `week_id`.
- Notebooks 3I and 3N use the common split metadata.
- Model-specific scaling is fitted inside each Optuna fold using that fold's training section only. Final scalers are fitted using the full training section only.
- Hyperparameter selection never reads the final test targets.
- Primary evaluation is **frozen fitted weights with rolling one-week-ahead forecasts**. The model parameters remain fixed across test weeks; after each week is observed, its actual count may be included in the historical lag/window for the next forecast. This gives Stream I and DCRNN the same information protocol.
- The DCRNN's Optuna budget is capped at 12 trials / 45 minutes, two temporal folds, at most 12 epochs per fold, patience 3, and at most 10 refit epochs. The final refit occurs after Optuna and is additional to the search timeout.

## Models

- `4IA`: `SGDRegressor(loss='squared_error')`
- `4IB`: `SGDRegressor(loss='huber')`
- `4IC`: `SGDRegressor(loss='epsilon_insensitive')`
- `4ID`: `MLPRegressor`
- `4N`: weekly state-graph DCRNN (PyTorch, GPU required)
- `5I`: equal-weight arithmetic mean of the four pretrained Stream I test predictions; it does not refit base models.

All Stream I model notebooks train on CPU using `partial_fit()` and Optuna. Every random state is 34 where the relevant API allows it. The DCRNN alone uses the GPU.

## Important Kaggle artifact workflow

Kaggle notebook sessions do not automatically share each notebook's `/kaggle/working` directory. Each notebook writes to:

`/kaggle/working/us_accidents_weekly_forecasting/`

To run these as separate Kaggle notebooks:

1. Run Notebook 1 and save its output/version.
2. Attach the saved output from Notebook 1 as an input to Notebook 2.
3. Repeat this for each notebook output needed by later notebooks. The `resolve_artifact()` helper searches `/kaggle/input/**` for required files.
4. If multiple older copies of an artifact are attached, set `ARTIFACT_INPUT_ROOT` near the top of the notebook to the intended input folder.
5. On a local Jupyter environment, the default shared directory is `./us_accidents_weekly_forecasting/`; running notebooks from the same working directory shares these artifacts.

Do not rename the artifact files unless you also update the corresponding `resolve_artifact()` calls.

## Main artifacts

Notebook 1:
- `artifacts/weekly_state_counts_sparse.parquet`
- `artifacts/source_aggregation_metadata.json`

Notebook 2:
- `artifacts/weekly_state_panel.parquet`
- `artifacts/weekly_national_counts.parquet`
- `artifacts/weekly_index.parquet`
- `artifacts/split_metadata.json`

Notebook 3I:
- `artifacts/stream_i_X_train_raw.npy`, `stream_i_y_train_raw.npy`
- `artifacts/stream_i_X_test_raw.npy`, `stream_i_y_test_raw.npy`
- Stream I train/test metadata and feature map

Notebook 3N:
- `artifacts/dcrnn_weekly_state_counts.npy`
- `artifacts/dcrnn_future_covariates.npy`
- `artifacts/dcrnn_adjacency.npy`
- DCRNN node mapping and preprocessing metadata

Notebooks 4IA–4ID:
- `models/*_bundle.joblib`
- `reports/*_test_predictions.csv`
- Optuna SQLite databases and trial CSVs

Notebook 4N:
- `models/weekly_dcrnn_state_graph.pt`
- `reports/dcrnn_test_predictions.csv`
- Per-state predictions, Optuna trials and DCRNN metadata

Notebook 5I:
- `reports/incremental_ensemble_test_predictions.csv`
- `reports/incremental_ensemble_manifest.json`

Notebook 6:
- consolidated regression metrics, merged predictions, and PNG visualizations

## Metric notes

The final scorecard reports MAE, RMSE, R², feature-count adjusted R², WAPE, Poisson deviance and sMAPE. `log_loss` is not suitable because it is a classification probability metric. Poisson deviance is included as a count-target deviance metric. Adjusted R² is only a supplementary feature-count correction and is not a true architecture-complexity correction for MLP/DCRNN/ensembles.

The first and last observed weeks are retained. Notebook 2 calculates `week_days_within_source_bounds` from the first/last valid source dates. This identifies only calendar truncation at the global dataset boundaries; it cannot detect internal collection gaps. The primary metrics exclude only test weeks truncated by those global boundaries. Partial boundary weeks remain in source-derived count tables and prediction outputs and are separately written to `partial_boundary_week_predictions.csv`.

## Verification status

The notebooks are programmatically generated and their structure/code syntax is checked in this bundle. They have **not** been run end-to-end against the 3 GB Kaggle CSV or Kaggle GPU runtime in this session because that dataset/runtime is not attached here. Run Notebook 1 on Kaggle first; it now fails fast if any timestamped row cannot be assigned to a week, and it checks row conservation before continuing.
