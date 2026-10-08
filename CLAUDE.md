# CLAUDE.md

## Conventions

### Kanban workflow

1. Work is tracked in the GitHub Project linked to this repo. Status columns: Backlog,
   Ready, In Progress, In Review, Done. WIP limit: at most 2 items In Progress.
2. Each roadmap phase is a GitHub Milestone ("Phase N - <name>"). Each phase is broken
   into small issues (half a day to two days of work), each with a type label, milestone,
   Priority and Size, and added to the project.
3. Before starting an issue: move it to In Progress and create branch
   `<type>/<issue-number>-<short-slug>` from an updated `main` (e.g., `feat/12-rent-scraper`).
4. When finished: run `make lint typecheck test`, update `CHANGELOG.md`, open a PR that
   says "Closes #<issue>", move the issue to In Review, then STOP, summarize the changes,
   and wait for my approval.
5. After approval: merge with a merge commit (never squash) so individual commits are
   preserved on `main`. Merged branches are deleted automatically. Never delete unmerged
   branches.
6. Extra work, bugs or ideas found along the way are not done silently: create a new issue
   in Backlog.
7. When every issue of a milestone is Done, close the milestone and create the tag/release
   if the roadmap says so.

### Code and data

8. Small commits with conventional commit messages (`feat:`, `fix:`, `docs:`, `test:`,
   `refactor:`, `chore:`, `ci:`, `data:`).
9. Dependencies only via `uv add` (dev tools with `--group dev`).
10. No notebooks. All logic, including exploratory and statistical analyses, lives in
    `src/` and runs with `make`; outputs go to `data/08_reporting` and conclusions to `docs/`.
11. Type hints and docstrings on all public functions; tests for every `src` module; target
    coverage >= 80%.
12. Everything in English: code, comments, docs, README, commits.
13. NEVER invent results. Every metric, number, or figure in the README must come from an
    actual run, saved in `data/08_reporting` (metrics as JSON/CSV, figures as PNG).
14. Data ethics: scraping only if robots.txt and terms of use allow it, with rate limiting
    and an identified user agent; never collect or publish personal data; do not commit
    data whose license forbids redistribution (commit the code plus a small sample
    instead); secrets only in `.env`, never committed (provide `.env.example`).

## Project

### Goal

Predict the monthly rent (COP) of apartments in Medellín, Colombia, and explain what drives
it: comuna/barrio, estrato, area (m²), rooms, bathrooms, parking, floor, building age and
amenities.

### Motivation

I'm moving into an unfurnished apartment in Medellín and want to know whether listed rents
are fair.

### Data

- **Listings:** rental listings from Colombian real-estate portals ONLY if robots.txt and
  the terms of use allow it; otherwise public datasets of Medellín listings. Availability
  and terms must be verified (issue #7) before any acquisition code is written.
- **Geography:** comunas and barrios boundaries from Medellín's official open data portal.
- Record every source, license and retrieval date in `data/README.md`.

### Domain notes

- **Comuna:** one of Medellín's urban administrative divisions; each contains barrios.
- **Barrio:** neighbourhood; listings usually name the barrio, which is mapped to its comuna.
- **Estrato:** socioeconomic stratum (1-6) assigned to residential properties in Colombia.
- Target: monthly rent in COP. Rent per m² is the main comparison unit in the analysis.

### Architecture

FTI pipelines in `src/medellin_rent/pipelines/`:

- **Feature** (`make feature`): `01_raw` -> `02_intermediate` -> `03_primary` -> `04_feature`
  -> `05_model_input`.
- **Training** (`make train`): `05_model_input` -> `06_models` + metrics/figures in
  `08_reporting`, tracked with MLflow.
- **Inference** (`make infer`, FastAPI): `06_models` + new data -> `07_model_output`.

### Stack

pandas, pandera, scikit-learn, LightGBM, MLflow, SHAP, geopandas, folium/plotly, FastAPI,
Docker, Streamlit.

### Roadmap

| Milestone | Scope |
|-----------|-------|
| Phase 0 - Setup | Repo, issues, Kanban board, CI green. |
| Phase 1 - Data acquisition | Acquisition module in `src` (scraper or loader) with tests; raw data in `01_raw`; data source documentation. |
| Phase 2 - Cleaning and validation | Pandera schemas; map barrios to comunas; outlier handling; outputs in `02` / `03`. |
| Phase 3 - EDA and statistical analysis | Price per m² by comuna and estrato, interactive map, hypothesis tests (e.g. effect of parking and estrato on price). |
| Phase 4 - Feature pipeline | Feature engineering and the feature pipeline. |
| Phase 5 - Training | Baseline (median by comuna), Ridge, Random Forest, LightGBM; k-fold CV; MLflow tracking; SHAP interpretation; metrics saved to `08_reporting`. |
| Phase 6 - Inference and API | Inference pipeline, FastAPI endpoint, Docker image. |
| Phase 7 - App and release | Streamlit app "Is this rent fair?" on Streamlit Community Cloud; README Results with real metrics and figures; release `v1.0.0`. |

### GitHub

- Repo: https://github.com/RafaGM1108/medellin-rent-predictor
- Kanban board: https://github.com/users/RafaGM1108/projects/11 (user project number `11`,
  fields Status, Priority, Size).
- Type labels: `type:feat`, `type:data`, `type:analysis`, `type:model`, `type:docs`,
  `type:test`, `type:ci`, `type:bug`.
- `main` is protected: changes land only through PRs with the `test` CI check passing.
