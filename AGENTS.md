# F1 Predictor — Agent Instructions

## Project Overview

Python ML application for Formula 1 race predictions. Uses `setuptools-scm` for versioning (derived from git tags) and Docker for deployment.

## Versioning & Releases

### How Versioning Works

- **Source of truth**: Git tags in `v{major}.{minor}.{patch}` format (e.g. `v0.1.0`)
- **`setuptools-scm`** reads the latest tag to set the Python package version at build time
- **Docker images** are tagged with semver via `docker/metadata-action` in `docker-publish.yml`

### Automatic Releases

Every merge to `main` automatically creates a **patch bump** via `.github/workflows/release.yml`.  
No action needed for bugfixes and minor changes.

### Manual Releases

For feature releases or breaking changes, use the GitHub Actions UI:

1. Go to **Actions** → **Release** → **Run workflow**
2. Select bump type: `patch`, `minor`, or `major`
3. The workflow creates a git tag, GitHub Release, and triggers Docker image publishing

### Skipping a Release

Add `[skip release]` to the merge commit message to skip automatic patch bumping.

## CI/CD Workflows

| Workflow | File | Triggers | Purpose |
|----------|------|----------|---------|
| Tests | `tests.yml` | PR to `main` | Runs pytest suite |
| Release | `release.yml` | Push to `main`, manual dispatch | Creates semver tag + GitHub Release |
| Docker | `docker-publish.yml` | Push to `main`, tag `v*.*.*`, PR | Builds and pushes Docker image to GHCR |

### Docker Image Tags

When a release tag is created, Docker images get these tags on `ghcr.io/2fst4u/f1predictor`:

- `v1.2.3` — exact version
- `1.2` — major.minor
- `sha-abc1234` — commit SHA
- `main` — latest from main branch

## Key Files

- `pyproject.toml` — Build config, setuptools-scm settings (`write_to = "f1pred/_version.py"`)
- `Dockerfile` — Multi-stage build; copies `.git/` for setuptools-scm version derivation
- `f1pred/util.py` — Reads version via `importlib.metadata.version("f1predictor")`
- `config.yaml` — Runtime configuration

## Testing

```bash
make test                              # Run full test suite
python -m pytest tests/test_release_config.py -v  # Validate release infrastructure
```

### Release Infrastructure Tests

`tests/test_release_config.py` enforces that release tooling stays consistent:
- setuptools-scm is configured in `pyproject.toml`
- Docker workflow has semver tag patterns
- Dockerfile copies `.git/` directory
- Release workflow exists with correct triggers

> [!IMPORTANT]
> If you modify `pyproject.toml`, `Dockerfile`, or any workflow file, run `test_release_config.py` to verify consistency.
