---
description: How to create a new release of f1predictor
---

# Releasing a New Version

## Prerelease (automatic — no action needed)

Every push to any branch automatically builds and pushes a Docker image to `ghcr.io/2fst4u/f1predictor` with a numerically increasing prerelease tag (e.g. `0.1.1-pre.42`). These are automatically picked up by Flux.

## Stable Release (manual)

1. Go to the GitHub repo → **Actions** tab
2. Select the **Release & Publish** workflow
3. Click **Run workflow**
4. Choose bump type:
   - `patch` — bugfixes (e.g. `v0.1.0` → `v0.1.1`)
   - `minor` — new features, backwards compatible (e.g. `v0.1.1` → `v0.2.0`)
   - `major` — breaking changes (e.g. `v0.2.0` → `v1.0.0`)
5. Click **Run workflow**

## What Happens After a Stable Release

1. `release.yml` creates an annotated git tag and GitHub Release with rolled-up release notes
2. Docker image is built and pushed to `ghcr.io/2fst4u/f1predictor` with semver tags (e.g. `0.2.0`, `0.2`)
3. `setuptools-scm` uses the tag for the Python package version
