# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Filled the Project section of `CLAUDE.md` with the spec and roadmap; renamed the workflow conventions to "Kanban workflow" (#3).
- Renamed the placeholder package `ds_project` to `medellin_rent` (#2).

### Added

- Initial project template: layered data folders, staged notebooks, `medellin_rent` package
  with feature/training/inference pipelines, FastAPI service and Streamlit app.
- Typed configuration loader for `conf/base.yaml` and shared logging setup.
- Tooling: uv, ruff, mypy, bandit, pytest + coverage, pre-commit, Makefile.
- GitHub Actions CI with Codecov upload, Dependabot, PR and issue templates.
- MkDocs Material documentation skeleton, devcontainer and multi-stage Dockerfile.
