# Methodology

## Data layers

Data flows `01_raw → 02_intermediate → 03_primary → 04_feature → 05_model_input →
06_models → 07_model_output → 08_reporting`. Raw data is immutable. See `data/README.md`.

## Pipelines

| Pipeline | Command | Reads | Writes |
|----------|---------|-------|--------|
| Feature | `make feature` | `01_raw` | `02_intermediate`, `03_primary`, `04_feature/features.parquet`, `05_model_input/{train,test}.parquet` |
| Training | `make train` | `05_model_input` | `06_models`, `08_reporting` |
| Inference | `make infer` | `06_models` + new data | `07_model_output` |

## Analyses

There are no notebooks. Exploratory and statistical analyses are scripts in `src/` run with
`make`; they write figures (PNG) and tables (CSV/JSON) to `data/08_reporting/`, and their
conclusions are written in `docs/`.
