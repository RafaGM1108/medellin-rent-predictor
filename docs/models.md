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

## Prediction interval

The 80% interval multiplies a prediction by the 10th and 90th percentiles of
`actual / predicted` rent on the test set: **0.78 to 1.26** (`test_metrics.json`). For a
predicted 2,000,000 COP that is roughly 1,565,000 to 2,515,000 COP. It is stored in
`model_metadata.json` and used by `make infer` and the API.

## What drives the predicted rent (SHAP, `make interpret`)

Source files: `shap_importance.csv`/`.png`, `shap_by_comuna.csv`, `shap_by_estrato.csv`/`.png`
and `shap_dependence_area.png`, computed for the saved LightGBM model on the 4,222 test
listings. SHAP values come from LightGBM's own TreeSHAP (`pred_contrib=True`), which handles
its native categories and missing values exactly. The model predicts `log(rent)`, so each
value is shown as a percentage change of the predicted rent, holding the other features at
the listing's values.

### Importance (mean absolute effect)

| Feature | Effect | Feature | Effect |
|---------|-------:|---------|-------:|
| comuna | 18.8% | estrato | 4.8% |
| bathrooms | 16.3% | area | 4.8% |
| longitude | 6.3% | furnished | 4.0% |
| latitude | 5.6% | parking | 2.7% |
| bedrooms | 5.1% | balcony | 2.3% |

The remaining features (listing quarter, gym, pool, doorman, elevator, area missing) each
move the prediction by 2% or less.

- **Location is the main driver**: comuna plus coordinates add up to the largest effects.
- **Bathrooms rank above area** because area is missing for about two thirds of listings;
  bathrooms and bedrooms then carry most of the information about size. Where the area is
  known, its effect rises steadily with size (`shap_dependence_area.png`): negative for small
  apartments, positive above roughly 100 m².
- **Being furnished matters more than any other amenity** (4.0%).

### Comuna (mean effect, comunas with at least 50 test listings)

| Comuna | Test listings | Effect |
|--------|-------------:|-------:|
| El Poblado | 1,441 | +29.0% |
| Laureles | 719 | -0.7% |
| Belén | 546 | -12.1% |
| La América | 377 | -14.2% |
| Buenos Aires | 252 | -17.0% |
| Robledo | 242 | -15.9% |
| La Candelaria | 200 | -17.2% |
| San Javier | 102 | -20.0% |
| San Cristóbal | 69 | -16.2% |
| Guayabal | 55 | -13.9% |

Effects are relative to the model's average prediction, which El Poblado's many listings
pull up; that is why most comunas are negative.

### Estrato (mean effect)

| Estrato | 1 | 2 | 3 | 4 | 5 | 6 | unknown |
|---------|--:|--:|--:|--:|--:|--:|--------:|
| Effect | -4.3% | -5.9% | -7.1% | -1.4% | +3.3% | +6.8% | -3.4% |
| Test listings | 17 | 102 | 756 | 997 | 1,036 | 1,174 | 140 |

- **Once location and size are known, estrato adds a smaller effect**: about 14 points
  between estrato 3 and 6, far less than the 61% raw gap in rent per m² (Phase 3). Much of
  the raw estrato effect is really location, as the comuna-by-estrato analysis suggested.
- Estratos 1 and 2 have few listings; their effects are not reliable.
