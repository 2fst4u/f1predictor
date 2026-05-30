# Browser end-to-end tests

These Playwright tests drive `f1pred/templates/index.html` in a real Chromium
instance to verify client-side (Alpine.js) behaviour that cannot be checked from
Python:

- `test_ui_fix.py` — background live-update round gating: an out-of-band
  prediction for a non-active round must be cached, not rendered, while an
  update matching the active round is rendered.
- `test_ui_layout.py` — responsive `getFactorLimit()` offsets across desktop /
  medium / mobile viewport widths.

## Why they live here and not in `tests/`

They require a browser, so they are **not** part of the default unit-test run
(`make test`, which only collects `tests/`). Keeping them under `e2e/` means the
default suite never silently skips them — they are an opt-in suite you run
explicitly when changing the UI.

## Running them

```bash
pip install pytest-playwright
playwright install chromium
pytest e2e/
```

If `pytest-playwright` is not installed the tests skip with a clear reason
rather than erroring.
