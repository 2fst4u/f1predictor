# F1 Predictor — Agent Instructions

## Project Overview

Python ML application for Formula 1 race predictions. Uses `setuptools-scm` for versioning (derived from git tags) and Docker for deployment.

## Versioning & Releases

### How Versioning Works

- **Source of truth**: Git tags in `v{major}.{minor}.{patch}` format (e.g. `v0.1.0`)
- **`setuptools-scm`** reads the latest tag to set the Python package version at build time
- **Prerelease images** use `{next-patch}-dev.{N}` format (numerically increasing for Flux)

### Prerelease Builds (automatic)

Every push to **any branch (except `main`)** triggers `build.yml` (ignoring changes to markdown and workflow files), which runs `Tests` (reusing `tests.yml`) → `Build`. Code review is **not** part of CI — it is invoked manually by commenting `/oc-review` on a PR or issue (handled by `opencode-review.yml`). This keeps CI fast and inexpensive; reviewers ask for a review only when wanted.

The build job produces a Docker image tagged with:

- `{next-patch}-dev.{DEV_NUM}` — numerically increasing (e.g. `0.1.1-dev.42`)
- `dev` — static tag that always points to the latest dev build
- Branch name (e.g. `feature-xyz`)
- Commit SHA (e.g. `sha-abc1234`)

The `dev` tag is automatically pulled by Flux for continuous deployment.

### Stable Releases (manual)

All stable releases are **manual** via GitHub Actions UI:

1. Go to **Actions** → **Release** → **Run workflow**
2. Select bump type: `patch`, `minor`, or `major`
3. The workflow creates a git tag, GitHub Release with rolled-up notes, and the published release automatically triggers a semver-tagged Docker image build

## CI/CD Workflows

| Workflow      | File                  | Triggers                                       | Purpose                                   |
| ------------- | --------------------- | ---------------------------------------------- | ----------------------------------------- |
| Tests         | `tests.yml`           | `workflow_call`                                | Reusable workflow to run pytest suite     |
| Build         | `build.yml`           | `push`, `workflow_call`, `release`             | Tests → Build Docker image                |
| Manual Review | `opencode-review.yml` | `issue_comment`, `pull_request_review_comment` | AI code review on demand via `/oc-review` |
| Release       | `release.yml`         | Manual dispatch only                           | Creates semver tag + GitHub Release       |

### Docker Image Tags

| Source                    | Tags on `ghcr.io/2fst4u/f1predictor`                |
| ------------------------- | --------------------------------------------------- |
| Prerelease (dev branches) | `0.1.1-dev.42`, `dev`, `branch-name`, `sha-abc1234` |
| Stable release (manual)   | `0.1.1`, `0.1`, `sha-abc1234`                       |

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
- Tests workflow runs as a reusable component via `workflow_call`
- Build workflow triggers on push (running tests first) and on release publication
- Build workflow does NOT trigger on `pull_request` and contains no `review` job
- `opencode-review.yml` exists and handles `/oc-review` comments via the OpenCode action
- No automatic review workflow runs on push or pull_request (reviews are strictly manual)
- Build workflow produces prerelease and semver Docker tags
- Release workflow is manual-only
- Old `docker-publish.yml` does not exist

> [!IMPORTANT]
> If you modify `pyproject.toml`, `Dockerfile`, or any workflow file, run `test_release_config.py` to verify consistency.

## Before you start: look at the open pull requests

List the open pull requests and check whether one already touches the file or
symbol you are about to change. If it does, work on something else.

Several agents run against this repository independently and cannot see each
other's work, so they converge on the same target often. One review across these
repositories found three byte-identical pull requests renaming a single import,
three separate attempts at splitting one function, and three sets of tests for
one endpoint — eleven of the thirty rejected pull requests were duplicates of
another open one.

`.github/workflows/overlapping-pr-check.yml` comments on a pull request when
another open one edits the same files. It does not block anything; overlap is
normal on a busy branch. Treat it as a prompt to check whether the two are doing
the same work, and to close the weaker one before both reach review.

## Reporting "no change needed"

Finding that a task needs no code change is a complete result. Report it and
stop. Do not open a pull request, and do not add an unrelated edit so that a
pull request has something to carry.

This used to be enforced the other way round. A CI gate (`empty-commit-check`)
failed any branch whose diff was empty. It did stop empty pull requests, and it
started a worse habit: agents began manufacturing a change to get the gate
green. One wrote the tactic into its own journal — *"apply a completely safe,
trivial code health cleanup ... to satisfy the CI file modification
requirement"* — directly beneath an older entry telling the same agent not to
introduce unnecessary modifications.

The filler was not always safe. Across one review of the open pull requests in
these repositories it suppressed worker error reporting in production builds,
made a failed upload report the wrong reason, and moved a content-type check to
after the response had been buffered into memory.

In its place, `.github/workflows/no-op-pr.yml` closes a pull request whose diff
is empty or touches only `.jules/**`, with a comment saying why. A close is not
a failure — it is the same finding, recorded without costing anyone a review.

A journal entry is not a change either. It records a lesson learned from work;
it belongs in the pull request carrying that work, or committed straight to the
default branch. On its own it is filler.
