# F1 Predictor — Agent Instructions

## Project Overview

Python ML application for Formula 1 race predictions. Uses `setuptools-scm` for versioning (derived from git tags) and Docker for deployment.

## Versioning & Releases

### How Versioning Works

- **Source of truth**: Git tags in `v{major}.{minor}.{patch}` format (e.g. `v0.1.0`)
- **`setuptools-scm`** reads the latest tag to set the Python package version at build time
- **Prerelease images** use `{next-patch}-pre.{N}` format (numerically increasing for Flux)

### Prerelease Builds (automatic)

Every push to **any branch** builds and pushes a Docker image tagged with:
- `{next-patch}-pre.{run_number}` — numerically increasing (e.g. `0.1.1-pre.42`)
- Branch name (e.g. `main`, `feature-xyz`)
- Commit SHA (e.g. `sha-abc1234`)

These are automatically pulled by Flux for continuous deployment.

### Stable Releases (manual)

All stable releases are **manual** via GitHub Actions UI:

1. Go to **Actions** → **Release & Publish** → **Run workflow**
2. Select bump type: `patch`, `minor`, or `major`
3. The workflow creates a git tag, GitHub Release with rolled-up notes, and builds a semver-tagged Docker image

## CI/CD Workflows

| Workflow | File | Triggers | Purpose |
|----------|------|----------|---------|
| Tests | `tests.yml` | PR to `main` | Runs pytest suite |
| Prerelease Build | `docker-publish.yml` | Push to any branch | Builds + pushes prerelease Docker image |
| Release & Publish | `release.yml` | Manual dispatch only | Creates semver tag, GitHub Release, + semver Docker image |

> [!IMPORTANT]
> Tags pushed by `GITHUB_TOKEN` do not trigger other workflows (GitHub limitation).
> This is why `release.yml` builds Docker images directly.

### Docker Image Tags

| Source | Tags on `ghcr.io/2fst4u/f1predictor` |
|--------|------|
| Prerelease (every push) | `0.1.1-pre.42`, `main`, `sha-abc1234` |
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
- Prerelease workflow pushes on every branch with incrementing versions
- Release workflow is manual-only with semver Docker tags

> [!IMPORTANT]
> If you modify `pyproject.toml`, `Dockerfile`, or any workflow file, run `test_release_config.py` to verify consistency.
