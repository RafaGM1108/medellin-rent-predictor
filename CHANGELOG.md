# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed

- `notebooks/`: analyses are reproducible scripts in `src/` run with `make`; CLAUDE.md convention, README, docs and tooling updated (#61).

### Changed

- Issue forms apply `type:feat` and `type:bug`, the labels used in this repo (#47).
- Dependabot no longer proposes major/minor Python bumps of the Docker base image, which must stay on 3.12 (#40).
- Codecov uploads authenticate with OIDC instead of a token secret and fail the CI on error; added `codecov.yml` with 80% project and patch targets (#6).
- Filled the README (overview, motivation, data, approach, roadmap, tech stack) and added a Project management section; removed the template setup section (#5).
- Split `data/01_raw` into `listings/` and `geo/`, added the `geo` module and documented the adaptation in the README (#4).
- Filled the Project section of `CLAUDE.md` with the spec and roadmap; renamed the workflow conventions to "Kanban workflow" (#3).
- Renamed the placeholder package `ds_project` to `medellin_rent` (#2).

### Added

- Training pipeline selects the best model by CV MAE, refits it, scores it once on the test set (`test_metrics.json`) and saves it to `06_models` with metadata; every model is an MLflow run in a local SQLite store (`paths.mlruns`); `docs/models.md`; `mlflow` dependency (#29).
- LightGBM (with and without target-encoded barrio) tuned by random search on inner folds of the training set; `lightgbm_tuning.csv`; `params.lightgbm_tuning_iter`; `lightgbm` dependency (#28).
- Ridge and Random Forest pipelines with area imputation by bedrooms fitted in the training folds; registry seeded from the config (#27).
- `medellin_rent.model` (shared k-fold CV and COP metrics, comuna-median baseline, model registry) and the training pipeline: `make train` writes `model_comparison.{csv,json}` and `model_cv_folds.csv`; `params.cv_folds`; `scikit-learn` dependency (#26).
- `docs/features.md`: feature dictionary with type, definition and source column of every feature and target (#25).
- Edge-case tests for features and model input: empty input, no listing with a comuna, duplicate listing ids (#24).
- Feature pipeline writes `04_feature/features.parquet` and `05_model_input/{train,test}.parquet` (listings with a comuna, `rent_cop` and `log_rent`, split with `params.test_size` and the project seed) (#23).
- `medellin_rent.features.build`: size, rooms, three-level parking, amenities, location (fixed comuna/estrato levels, lat/lon, barrio) and listing quarter features, computed per listing with no data-derived statistics (#22).
- `docs/decisions.md`: key findings of Phases 1-3 and the modelling decisions for Phases 4-5 (target, rows, features, validation, baseline, price adjustment) (#21).
- `medellin_rent.analysis.rent_map`: choropleth of median rent per m² by comuna, static PNG (committed, in the README) and interactive folium HTML (git-ignored because it embeds the boundaries); `folium` dependency (#19).
- `medellin_rent.analysis.hypothesis`: rank-based tests of the effect of parking (overall and within estrato) and of estrato on rent, with effect sizes, bootstrap CIs and Holm adjustment; `hypothesis_tests.json` and conclusions in `docs/analysis.md`; `scipy` dependency (#20).
- `medellin_rent.analysis.price_m2`: rent per m² by comuna, by estrato and by both (CSV + PNG in `data/08_reporting/`), with conclusions in `docs/analysis.md` (#18).
- `medellin_rent.analysis` and `make analysis`: EDA tables and figures in `data/08_reporting/` (distributions, data quality) and `docs/analysis.md`; `matplotlib` dependency (#17).
- Estrato from the official GeoMedellín estrato layer for listings with coordinates (`assign_estrato`, `estrato_source`); `make geo` downloads the layer (#59).
- Edge-case tests for cleaning (inclusive thresholds, missing values, empty input) and location mapping (overlapping polygons, empty input) (#16).
- `medellin_rent.data.clean` and the primary step of the feature pipeline: duplicate, rent, area, count and floor rules with logged effects; located, cleaned and validated `03_primary/listings.parquet`; `text_hash` column for duplicate detection without the text (#15).
- `medellin_rent.geo.mapping`: barrio and comuna for each listing from the declared comuna name and a point-in-polygon join, with name normalization, conflict handling and logged unmatched names (#14).
- Pandera schemas for the intermediate and primary listings tables; the feature pipeline validates the intermediate table; `pandera` dependency (#13).
- `medellin_rent.data.parse` and the first feature pipeline step: typed intermediate table in `02_intermediate/listings.parquet` with area, rooms, bathrooms, parking, floor, age, estrato and amenities extracted from the text when not structured; text columns dropped; `pyarrow` dependency (#12).
- Data documentation: sources, licenses, collection steps, what is committed and ethics (`docs/data.md`) (#11).
- Offline tests for the listings loader on a synthetic fixture in the raw file format: multiline text, empty values, filters, redaction edge cases and errors (#9).
- `medellin_rent.data.listings` and `make listings`: load Medellín apartment rentals from the Properati file with phone numbers and emails redacted, and record the file's checksum; `pandas` dependency (#8).
- `medellin_rent.geo.boundaries` and `make geo`: download the official comuna and barrio boundaries with source metadata and load them in EPSG:4326; `geopandas` dependency (#10).
- Data source survey: robots.txt and terms of use of five portals, public datasets and Medellín open data layers; decision not to scrape (#7).
- Initial project template: layered data folders, staged notebooks, `medellin_rent` package
  with feature/training/inference pipelines, FastAPI service and Streamlit app.
- Typed configuration loader for `conf/base.yaml` and shared logging setup.
- Tooling: uv, ruff, mypy, bandit, pytest + coverage, pre-commit, Makefile.
- GitHub Actions CI with Codecov upload, Dependabot, PR and issue templates.
- MkDocs Material documentation skeleton, devcontainer and multi-stage Dockerfile.
