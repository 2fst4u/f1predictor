---
description: How to create a new release of f1predictor
---

# Releasing a New Version

Versions are managed automatically. Every merge to `main` creates a patch bump. For manual releases:

## Auto Patch (default — no action needed)

Merging a PR to `main` automatically bumps the patch version (e.g. `v0.1.0` → `v0.1.1`).

To skip, include `[skip release]` in the merge commit message.

## Manual Minor/Major Release

1. Go to the GitHub repo → **Actions** tab
2. Select the **Release** workflow
3. Click **Run workflow**
4. Choose bump type:
   - `minor` — new features, backwards compatible (e.g. `v0.1.1` → `v0.2.0`)
   - `major` — breaking changes (e.g. `v0.2.0` → `v1.0.0`)
   - `patch` — bugfixes (same as auto)
5. Click **Run workflow**

## What Happens After a Release

1. `release.yml` creates an annotated git tag and GitHub Release
2. The tag push triggers `docker-publish.yml`
3. Docker image is built and pushed to `ghcr.io/2fst4u/f1predictor` with semver tags
4. `setuptools-scm` uses the tag for the Python package version

## CLI Alternative

```bash
# Create tag manually
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

This also triggers the Docker publish workflow but does NOT create a GitHub Release automatically.
