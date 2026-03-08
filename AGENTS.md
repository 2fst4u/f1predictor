# F1 Predictor — Agent Instructions

## Project Overview

Python ML application for Formula 1 race predictions. Uses `setuptools-scm` for versioning (derived from git tags) and Docker for deployment.

## Versioning & Releases

### How Versioning Works

- **Source of truth**: Git tags in `v{major}.{minor}.{patch}` format (e.g. `v0.1.0`)
- **`setuptools-scm`** reads the latest tag to set the Python package version at build time
- **Prerelease images** use `{next-patch}-pre.{N}` format (numerically increasing for Flux)

### Prerelease Builds (automatic)

Every push to **any non-main branch**, or direct push to **main**, triggers a build (merges to main are skipped to avoid redundancy). The job first runs tests, and if successful, builds a Docker image tagged with:
- `{next-patch}-pre.{run_number}` — numerically increasing (e.g. `0.1.1-pre.42`)
- `prerelease` — static tag that always points to the latest dev build
- Branch name (e.g. `main`, `feature-xyz`)
- Commit SHA (e.g. `sha-abc1234`)

The `prerelease` tag is automatically pulled by Flux for continuous deployment.

### Stable Releases (manual)

All stable releases are **manual** via GitHub Actions UI:

1. Go to **Actions** → **Release** → **Run workflow**
2. Select bump type: `patch`, `minor`, or `major`
3. The workflow creates a git tag, GitHub Release with rolled-up notes, and the published release automatically triggers a semver-tagged Docker image build

## CI/CD Workflows

| Workflow | File | Triggers | Purpose |
|----------|------|----------|---------|
| Tests | `tests.yml` | `push`, `workflow_call` | Runs pytest suite on every commit |
| Build | `build.yml` | `workflow_run` (main), `pull_request`, `release` | Runs tests, then builds + pushes Docker image |
| Release | `release.yml` | Manual dispatch only | Creates semver tag + GitHub Release |

### Docker Image Tags

| Source | Tags on `ghcr.io/2fst4u/f1predictor` |
|--------|------|
| Prerelease (dev branches & direct main pushes) | `0.1.1-pre.42`, `prerelease`, `branch-name`, `sha-abc1234` |
| Stable release (manual) | `0.1.1`, `0.1`, `sha-abc1234` |

## Key Files

- `pyproject.toml` — Build config, setuptools-scm settings (`write_to = "f1pred/_version.py"`)
- `Dockerfile` — Multi-stage build; copies `.git/` for setuptools-scm version derivation
- `f1pred/util.py` — Reads version via `importlib.metadata.version("f1predictor")`
- `config.yaml` — Runtime configuration

## Testing

```bash
python -m pytest --cov=f1pred tests/ -v  # Run full test suite
python -m pytest tests/test_release_config.py -v  # Validate release infrastructure
```

### Release Infrastructure Tests

`tests/test_release_config.py` enforces that release tooling stays consistent:
- setuptools-scm is configured in `pyproject.toml`
- Dockerfile copies `.git/` directory
- Tests workflow runs on push and as a reusable component via `workflow_call`
- Build workflow triggers on workflow_run (main) or pull_request (running tests first) and on release publication
- Build workflow produces prerelease and semver Docker tags
- Release workflow is manual-only
- Old `docker-publish.yml` does not exist

> [!IMPORTANT]
> If you modify `pyproject.toml`, `Dockerfile`, or any workflow file, run `test_release_config.py` to verify consistency.
