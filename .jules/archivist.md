## 2024-03-10 - Undocumented Configuration Drift
**Learning:** Configuration options often get added to `config.yaml` and validation logic (`f1pred/config.py`) without corresponding updates in `README.md`, causing documentation to drift from actual supported behavior.
**Action:** When auditing configuration options, cross-reference `config.yaml` values explicitly against the `README.md` to spot undocumented parameters (like `precipitation_unit`).

## 2024-03-10 - Test Dependency Drift
**Learning:** The development environment requires manual installation of test and runtime tools (`playwright`, `fastparquet`, `pyarrow`, `colorama`) beyond the base dependencies in `requirements.txt`. The `README.md` instructions frequently drift from the actual required sequence.
**Action:** When updating test instructions, ensure the sequence explicitly includes `make install` followed by the manual installation of all test dependencies before running `make test`.
