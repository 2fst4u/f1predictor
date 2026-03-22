## 2024-03-10 - Undocumented Configuration Drift
**Learning:** Configuration options often get added to `config.yaml` and validation logic (`f1pred/config.py`) without corresponding updates in `README.md`, causing documentation to drift from actual supported behavior.
**Action:** When auditing configuration options, cross-reference `config.yaml` values explicitly against the `README.md` to spot undocumented parameters (like `precipitation_unit`).

## 2024-03-10 - Test Dependency Drift
**Learning:** The development environment requires manual installation of test and runtime tools (`playwright`, `fastparquet`, `pyarrow`, `colorama`) beyond the base dependencies in `requirements.txt`. The `README.md` instructions frequently drift from the actual required sequence.
**Action:** When updating test instructions, ensure the sequence explicitly includes `make install` followed by the manual installation of all test dependencies before running `make test`.

## 2026-03-22 - Variable Glossary Drift
**Learning:** Model features and their internal keys (`form_index`, `teammate_delta`, etc.) frequently evolve in `f1pred/features.py` and `f1pred/predict.py` without corresponding updates in `README.md`, causing documentation to become vague or outdated.
**Action:** When modifying feature engineering or model input logic, always update the 'Input Variables' section in `README.md` to ensure internal keys are correctly mapped to their human-readable descriptions.
