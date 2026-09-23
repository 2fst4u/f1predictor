## 2024-03-10 - Undocumented Configuration Drift

**Learning:** Configuration options often get added to `config.yaml` and validation logic (`f1pred/config.py`) without corresponding updates in `README.md`, causing documentation to drift from actual supported behavior.
**Action:** When auditing configuration options, cross-reference `config.yaml` values explicitly against the `README.md` to spot undocumented parameters (like `precipitation_unit`).

## 2024-03-10 - Test Dependency Drift

**Learning:** The development environment requires manual installation of test and runtime tools (`pytest-playwright`, `pytest-asyncio`, `fastparquet`, `pyarrow`, `colorama`) beyond the base dependencies in `requirements.txt`. The `README.md` instructions frequently drift from the actual required sequence, specifically omitting the browser binary installation required by `pytest-playwright` and missing `pytest-playwright` and `pytest-asyncio` themselves. Missing `pytest-playwright` but having `playwright` installed crashes local tests with a missing `page` fixture error.
**Action:** When updating test instructions, ensure the sequence explicitly includes `make install` followed by the manual installation of all test dependencies (including `pytest-playwright`, `pytest-asyncio`, and `playwright install --with-deps chromium`) before running `make test`.

## 2026-03-22 - Variable Glossary Drift

**Learning:** Model features and their internal keys (`form_index`, `teammate_delta`, etc.) frequently evolve in `f1pred/features.py` and `f1pred/predict.py` without corresponding updates in `README.md`, causing documentation to become vague or outdated.
**Action:** When modifying feature engineering or model input logic, always update the 'Input Variables' section in `README.md` to ensure internal keys are correctly mapped to their human-readable descriptions.

## 2024-04-22 - Feature Label Documentation Drift

**Learning:** Model features in `_FEATURE_LABELS` (like `sprint_form_index`, `is_sprint`) and deprecated features (like `team_tenure_events`) frequently drift between `f1pred/predict.py` and `README.md`.
**Action:** When adding or removing features from the backend or UI templates, ensure the 'Input Variables' section in `README.md` is explicitly synchronized to prevent documentation drift and orphaned UI labels.

## 2024-05-24 - Documenting Weather Features

**Learning:** Weather condition sensitivity features (`weather_beta_temp`, `weather_beta_pressure`, `weather_beta_wind`, `weather_beta_rain`) were added to the codebase but left undocumented in the README.md's "Input Variables" section, creating drift between the feature list and the actual implementation.
**Action:** When adding new features or sensitivity parameters to `f1pred/predict.py`, always ensure they are fully documented in the `README.md` to keep the model's inputs fully transparent.

## 2024-06-15 - Undocumented Configuration Drift

**Learning:** Application-level configuration options defined in `f1pred/config.py` and `config.yaml` (such as `live_refresh_seconds`, `auto_refresh_seconds`, `timezone`, and `log_level`) frequently drift from the documented setup in `README.md`, causing users to be unaware of tweakable application parameters.
**Action:** When auditing or modifying application configurations, always explicitly cross-reference and update the `README.md` Configuration section to prevent undocumented configuration drift.

## 2026-06-10 - Undocumented Feature Flags in Configuration

**Learning:** Feature toggles in the `modelling.features` section of `config.yaml` (such as `include_fastf1_fill`, `include_circuit_elevation`, and `include_weather_ensemble`) often drift from the `README.md` documentation. When new modeling features are introduced, developers update `config.py` and `config.yaml` but forget to document these flags in the `README.md` Configuration section.
**Action:** When adding new feature flags or modeling configurations, always update the Configuration section in `README.md` to match the actual `config.yaml` defaults and provide visibility to users.

## 2026-07-29 - Feature Flag Documentation Drift

**Learning:** The feature flags in `config.yaml` (`include_fastf1_fill`, `include_circuit_elevation`, `include_weather_ensemble`) were present but missing from the Configuration section in `README.md`.
**Action:** Always verify that newly added feature flags or modelling variables in `config.yaml` are also documented accurately in `README.md`.

## 2024-07-08 - Docker Tag Documentation Drift

**Learning:** The `AGENTS.md` file incorrectly stated that dev builds use `-pre` and `prerelease` tags, while the actual GitHub Actions workflow (`build.yml`) generates `-dev` and `dev` tags. The documentation drifted from the CI configuration.
**Action:** When inspecting CI/CD or deployment documentation, explicitly cross-reference the stated artifact tags against the actual workflow YAML files (like `build.yml`) to ensure consistency.

## 2026-07-29 - CI Trigger Documentation Drift

**Learning:** `AGENTS.md` incorrectly claimed that dev builds run on "any branch" (including `main`), but the CI configuration (`build.yml`) explicitly ignores the `main` branch. This led to inaccurate documentation about what triggers automatic prerelease builds and what branch tags exist.
**Action:** Always verify GitHub Actions trigger conditions (like `branches-ignore`) in `.github/workflows/` before trusting written claims in documentation about CI execution.

## 2026-08-26 - Configuration Documentation Drift

**Learning:** The configuration for data sources in `config.yaml` frequently drifts from what is documented in `README.md`. Specifically, the `jolpica` and `fastf1` `enabled` toggles were entirely omitted from the README's `data_sources` section.
**Action:** When updating the application configuration or reviewing documentation, ensure that all sub-blocks under `data_sources` in `config.yaml` are explicitly listed in the `README.md` Configuration section so users know how to toggle API dependencies.

## 2026-09-02 - Configuration Drift for open_meteo

**Learning:** The `enabled` configuration flag for `data_sources.open_meteo` was documented in the codebase (`config.py` and `config.yaml`) but mistakenly omitted from the Configuration example section in `README.md`.
**Action:** When adding or modifying data source configurations, always cross-reference the toggles against the `README.md` Configuration section to ensure no `enabled` flags are left undocumented.

## 2026-10-15 - Workflow Documentation Drift in .agents

**Learning:** Documentation located in `.agents/workflows/` (like `releasing.md`) frequently drifts from the actual GitHub Actions YAML configurations in `.github/workflows/`. Specifically, branch filtering rules (`branches-ignore`) and generated artifact tags (e.g., `-pre` vs `-dev`) were outdated compared to reality.
**Action:** When updating or reviewing documentation related to CI/CD and deployment processes (especially inside `.agents/`), explicitly cross-reference statements with the actual workflow files (`.github/workflows/*.yml`) to ensure branch rules and artifact tags match reality.
