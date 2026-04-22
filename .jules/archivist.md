## 2024-03-10 - Undocumented Configuration Drift
**Learning:** Configuration options often get added to `config.yaml` and validation logic (`f1pred/config.py`) without corresponding updates in `README.md`, causing documentation to drift from actual supported behavior.
**Action:** When auditing configuration options, cross-reference `config.yaml` values explicitly against the `README.md` to spot undocumented parameters (like `precipitation_unit`).

## 2024-03-10 - Test Dependency Drift
**Learning:** The development environment requires manual installation of test and runtime tools (`playwright`, `fastparquet`, `pyarrow`, `colorama`) beyond the base dependencies in `requirements.txt`. The `README.md` instructions frequently drift from the actual required sequence, specifically omitting the browser binary installation required by `pytest-playwright`.
**Action:** When updating test instructions, ensure the sequence explicitly includes `make install` followed by the manual installation of all test dependencies (including `playwright install --with-deps chromium`) before running `make test`.

## 2026-03-22 - Variable Glossary Drift
**Learning:** Model features and their internal keys (`form_index`, `teammate_delta`, etc.) frequently evolve in `f1pred/features.py` and `f1pred/predict.py` without corresponding updates in `README.md`, causing documentation to become vague or outdated.
**Action:** When modifying feature engineering or model input logic, always update the 'Input Variables' section in `README.md` to ensure internal keys are correctly mapped to their human-readable descriptions.

## 2024-04-22 - Feature Label Documentation Drift
**Learning:** Model features in `_FEATURE_LABELS` (like `sprint_form_index`, `is_sprint`) and deprecated features (like `team_tenure_events`) frequently drift between `f1pred/predict.py` and `README.md`.
**Action:** When adding or removing features from the backend or UI templates, ensure the 'Input Variables' section in `README.md` is explicitly synchronized to prevent documentation drift and orphaned UI labels.
