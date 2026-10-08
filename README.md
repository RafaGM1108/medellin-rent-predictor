# Medellín Rent Predictor

[![CI](https://github.com/RafaGM1108/medellin-rent-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/RafaGM1108/medellin-rent-predictor/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/RafaGM1108/medellin-rent-predictor/graph/badge.svg)](https://codecov.io/gh/RafaGM1108/medellin-rent-predictor)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> Is this rent fair? Predicting and explaining monthly apartment rents in Medellín, Colombia.

## Overview

This project predicts the monthly rent (COP) of an apartment in Medellín from its
characteristics (comuna/barrio, estrato, area in m², rooms, bathrooms, parking, floor,
building age and amenities) and explains which of them drive the price. The model is served
through a FastAPI endpoint and a Streamlit app that answers one question: *is this listed
rent fair?*

![Median rent per m² by comuna](data/08_reporting/rent_map_comunas.png)

*Median rent per m² by comuna, from `make analysis` (details in [docs/analysis.md](docs/analysis.md)). Boundaries: Alcaldía de Medellín (GeoMedellín), CC BY-SA 4.0.*

## Motivation

I'm moving into an unfurnished apartment in Medellín and wanted to know whether listed rents
are fair. Asking prices vary widely between comunas, estratos and buildings, and it is hard
to tell from a few listings whether a price is reasonable. A model trained on many listings,
with interpretable drivers, gives a reference price and the reasons behind it.

## Data

- **Rental listings.** Colombian real-estate portals are used **only** if their robots.txt
  and terms of use allow it, with rate limiting and an identified user agent; otherwise a
  public dataset of Medellín listings is used. No personal data is collected. The source
  survey and the decision are tracked in
  [#7](https://github.com/RafaGM1108/medellin-rent-predictor/issues/7).
- **Geographic layers.** Comunas and barrios boundaries from Medellín's official open data
  portal, used to map each listing to its comuna and to draw maps.

Sources, licenses and retrieval dates are documented in [`data/README.md`](data/README.md).
Data whose license forbids redistribution is not committed; only code and small samples are.

## Approach

The project follows a feature / training / inference (FTI) pipeline architecture.

1. **Acquisition.** A scraper or loader in `src/` stores listings and boundaries unchanged in
   `data/01_raw/`.
2. **Cleaning and validation.** Listings are parsed, validated with pandera schemas, mapped
   from barrio to comuna and cleaned of duplicates and outliers (`02_intermediate`,
   `03_primary`).
3. **Exploration and statistics.** Rent per m² by comuna and estrato, an interactive map and
   hypothesis tests (e.g. the effect of parking and estrato on price).
4. **Feature pipeline.** Reusable feature functions build `04_feature` and `05_model_input`.
5. **Training pipeline.** A baseline (median by comuna), Ridge, Random Forest and LightGBM
   are compared with k-fold cross-validation and tracked in MLflow; the best model is
   interpreted with SHAP. Metrics and figures are saved to `08_reporting`.
6. **Inference and serving.** An inference pipeline, a FastAPI `/predict` endpoint in a
   Docker image, and a Streamlit app deployed on Streamlit Community Cloud.

## Results

TODO: results go here **only** after an actual run. Every number and figure must come
from a file in [`data/08_reporting/`](data/08_reporting/).

| Metric | Value | Source |
|--------|-------|--------|
| TODO | TODO | `data/08_reporting/TODO.json` |

## Project structure

```text
├── conf/base.yaml          # Paths and parameters
├── data/
│   ├── 01_raw/
│   │   ├── listings/       # Rental listings as acquired (immutable)
│   │   └── geo/            # Comunas/barrios boundaries from Medellín open data
│   ├── 02_intermediate/ … 07_model_output/
│   └── 08_reporting/       # Metrics (JSON/CSV) and figures (PNG) behind the Results
├── docs/                   # MkDocs Material site
├── src/medellin_rent/
│   ├── data/               # Listings acquisition (scraper or loader) and cleaning
│   ├── geo/                # Boundaries loading and barrio → comuna mapping
│   ├── analysis/           # Reproducible analyses (make analysis)
│   ├── features/           # Feature engineering
│   ├── model/              # Training and evaluation
│   ├── inference/          # Prediction logic
│   ├── pipelines/          # feature_pipeline, training_pipeline, inference_pipeline
│   ├── api/                # FastAPI service
│   ├── app/                # Streamlit app ("Is this rent fair?")
│   └── utils/              # Config loader and logging
├── tests/                  # Mirrors src/
├── Dockerfile              # Multi-stage image for the API
└── Makefile                # Common commands (make help)
```

### Adaptation from the template

This repo was created from
[ds-project-template](https://github.com/RafaGM1108/ds-project-template). The layout was
adapted to the two kinds of data this project combines:

- **`data/01_raw/` is split into `listings/` and `geo/`.** Listings and geographic
  boundaries come from different sources, with different licenses, formats and refresh
  cycles, so each keeps its own raw folder. Both are set in `conf/base.yaml`
  (`paths.raw_listings`, `paths.raw_geo`).
- **New `src/medellin_rent/geo/` module** (tested in `tests/geo/`). Loading the official
  comunas/barrios boundaries and mapping each listing's barrio to its comuna is spatial
  logic used by both the cleaning stage and the app, so it lives apart from `data/`.
- **`src/medellin_rent/data/`** holds the listings acquisition module (scraper or loader,
  decided in Phase 1 after checking robots.txt and terms of use) and the cleaning code.

- **No notebooks.** The template's `notebooks/` folder was removed: every analysis is a
  script in `src/` run with `make`, so figures and tables in `data/08_reporting/` can be
  regenerated from the data at any time.

Everything else follows the template: the layered data folders and FTI pipelines.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and `make`.

```bash
git clone https://github.com/RafaGM1108/medellin-rent-predictor.git
cd medellin-rent-predictor
make install                 # dependencies + pre-commit hooks
cp .env.example .env         # then fill in secrets, if any
make lint typecheck test     # check everything works
```

| Command | What it does |
|---------|--------------|
| `make geo` | Download comuna, barrio and estrato layers into `data/01_raw/geo/` |
| `make listings` | Check the manually downloaded listings file and record its checksum |
| `make feature` / `make train` / `make infer` | Run the pipelines |
| `make analysis` | Regenerate the analysis figures and tables in `data/08_reporting/` |
| `uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db` | Browse the training runs |
| `make api` | FastAPI on http://localhost:8000 (docs at `/docs`) |
| `make app` | Streamlit on http://localhost:8501 |
| `make coverage` | Tests with an HTML coverage report |
| `make docs` | Serve the documentation locally |

Docker (API):

```bash
docker build -t medellin-rent-predictor .
docker run --rm -p 8000:8000 medellin-rent-predictor
```

## Roadmap

| Phase | Milestone | Scope | Status |
|-------|-----------|-------|--------|
| 0 | [Phase 0 - Setup](https://github.com/RafaGM1108/medellin-rent-predictor/milestone/1) | Repo, issues, Kanban board, CI green | ✅ Done |
| 1 | [Phase 1 - Data acquisition](https://github.com/RafaGM1108/medellin-rent-predictor/milestone/2) | Acquisition module (scraper or loader) with tests; raw data; source documentation | ✅ Done |
| 2 | [Phase 2 - Cleaning and validation](https://github.com/RafaGM1108/medellin-rent-predictor/milestone/3) | Pandera schemas, barrio → comuna mapping, outlier handling | ✅ Done |
| 3 | [Phase 3 - EDA and statistical analysis](https://github.com/RafaGM1108/medellin-rent-predictor/milestone/4) | Price per m² by comuna and estrato, interactive map, hypothesis tests | ✅ Done |
| 4 | [Phase 4 - Feature pipeline](https://github.com/RafaGM1108/medellin-rent-predictor/milestone/5) | Feature engineering and the feature pipeline | ✅ Done |
| 5 | [Phase 5 - Training](https://github.com/RafaGM1108/medellin-rent-predictor/milestone/6) | Baseline, Ridge, Random Forest, LightGBM; k-fold CV; MLflow; SHAP | 🟡 In progress |
| 6 | [Phase 6 - Inference and API](https://github.com/RafaGM1108/medellin-rent-predictor/milestone/7) | Inference pipeline, FastAPI endpoint, Docker image | ⬜ |
| 7 | [Phase 7 - App and release](https://github.com/RafaGM1108/medellin-rent-predictor/milestone/8) | Streamlit app on Streamlit Community Cloud, README results, `v1.0.0` | ⬜ |

## Project management

Work is planned and tracked on a public Kanban board:
**[medellin-rent-predictor · Kanban](https://github.com/users/RafaGM1108/projects/11)**.

- **Columns:** Backlog → Ready → In Progress → In Review → Done, with at most 2 items
  In Progress at a time.
- **Milestones:** each roadmap phase is a milestone (`Phase N - <name>`) broken into small
  issues of half a day to two days of work.
- **Issues:** each has a description, an acceptance criteria checklist, a type label
  (`type:feat`, `type:data`, `type:analysis`, `type:model`, `type:docs`, `type:test`,
  `type:ci`, `type:bug`), a milestone, and **Priority** (High/Medium/Low) and **Size**
  (S/M/L) fields on the board.
- **Flow:** one issue → one branch (`<type>/<issue>-<slug>`) → one PR that says
  `Closes #<issue>`. `main` is protected: PRs need the CI check to pass and are merged with a
  merge commit, so every commit is kept. Work found along the way becomes a new Backlog issue.
- **Changes** are recorded in [`CHANGELOG.md`](CHANGELOG.md); phases that ship a release are
  tagged.

## Tech stack

| Area | Tools |
|------|-------|
| Data | pandas, pandera, geopandas |
| Modelling | scikit-learn, LightGBM, SHAP, MLflow |
| Visualisation | plotly, folium |
| Serving | FastAPI, Docker, Streamlit (Community Cloud) |
| Engineering | Python 3.12, uv, ruff, mypy, bandit, pytest, pre-commit, GitHub Actions, Codecov, MkDocs Material |

Libraries are added with `uv add` in the phase that first needs them.

## Author

**Rafael E. Garcia**

- GitHub: [@RafaGM1108](https://github.com/RafaGM1108)
- LinkedIn: [linkedin.com/in/rafaelgarcia11](https://www.linkedin.com/in/rafaelgarcia11)

Licensed under the [MIT License](LICENSE).
