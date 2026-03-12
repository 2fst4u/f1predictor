## 2024-03-04 - Remove leftover performance testing and debug scripts
**Learning:** Performance test scripts (e.g., `test_perf*.py`) and small debug test runners (e.g., `test_output_format.py`) created during local performance optimizations can sometimes be accidentally committed to the repository root. These do not belong in the formal `tests/` directory and are not executed by the test suite.
**Action:** When acting as Sweep, look out for standalone scripts matching `test_*.py` located in the root directory rather than the formal `tests/` directory, as these are often orphaned debug artifacts and safe to remove after verification.

## 2025-03-12 - Remove leftover test_cache artifact
**Learning:** Local test execution generates cache directories (e.g., `test_cache/predictions`, `test_cache/fastf1/fastf1_http_cache.sqlite`) which must be explicitly removed or ignored before committing to avoid introducing repository bloat.
**Action:** Always add `test_cache/` to `.gitignore` when cleaning up leftover test cache files to prevent re-committing them in the future.
