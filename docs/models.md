# Models

Produced by `make train` from `05_model_input` (16,890 training and 4,222 test listings).
Numbers come from `data/08_reporting/model_comparison.csv`, `lightgbm_tuning.csv` and
`test_metrics.json`; runs are also tracked in MLflow
(`uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`).

## Setup

- **Target**: `log(rent)`; predictions are turned back into COP before scoring.
- **Cross-validation**: the same 5 shuffled folds (project seed) for every model.
- **Tuning**: LightGBM random search over 20 configurations, scored on 3 inner folds of the
  training set drawn with a different seed. The test set is used once, for the selected
  model.
- **Selection**: the model with the lowest mean CV MAE is refit on the whole training set,
  scored on the test set and saved to `06_models/model.joblib` (with `model_metadata.json`).

## Cross-validation (5 folds, mean ± standard deviation)

| Model | MAE (COP) | RMSE (COP) | MAPE |
|-------|----------:|-----------:|-----:|
| LightGBM | 374,166 ± 8,595 | 866,113 ± 6,710 | 16.0% ± 0.4 |
| LightGBM + barrio target encoding | 380,277 ± 11,863 | 875,058 ± 16,794 | 16.4% ± 0.4 |
| Random Forest | 399,000 ± 9,420 | 920,036 ± 23,299 | 17.0% ± 0.3 |
| Ridge | 512,155 ± 11,767 | 1,065,520 ± 33,146 | 23.3% ± 0.3 |
| Baseline (comuna median) | 646,954 ± 18,792 | 1,292,196 ± 70,485 | 29.8% ± 0.5 |

## Selected model on the test set

**LightGBM** (num_leaves 127, learning_rate 0.1, n_estimators 600, min_child_samples 10,
subsample 1.0, colsample_bytree 0.8, reg_lambda 0), 4,222 held-out listings:

| MAE (COP) | RMSE (COP) | MAPE |
|----------:|-----------:|-----:|
| 351,516 | 816,338 | 15.6% |

## Reading the results

- **LightGBM roughly halves the baseline's error in relative terms**: MAPE 16.0% against
  29.8% in CV.
- **Barrio target encoding does not help**: the coordinates already carry that information.
- **RMSE is more than twice the MAE** for every model: a few expensive listings have large
  errors. The model is least reliable at the top of the market.
- **Limits.** The tuned LightGBM's CV score is slightly optimistic because tuning and
  evaluation use the same rows; the test score is the honest estimate. The best
  configuration sits at the edge of the search space for `num_leaves` and `learning_rate`.
  Prices are from 2020-2021 (see #67).
