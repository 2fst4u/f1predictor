"""Tests to validate that release infrastructure files stay consistent.

These are pure file-parsing tests (no network, no Docker) that catch accidental
breakage of the version/release/Docker toolchain.
"""
from pathlib import Path

import yaml

# All paths relative to project root
ROOT = Path(__file__).resolve().parent.parent


class TestReleaseInfrastructure:
    """Validate that release tooling stays consistent across config files."""

    def test_setuptools_scm_configured(self):
        """pyproject.toml must have setuptools-scm in build requirements."""
        pyproject = ROOT / "pyproject.toml"
        assert pyproject.exists(), "pyproject.toml not found"
        content = pyproject.read_text()
        assert "setuptools-scm" in content, (
            "setuptools-scm must be in pyproject.toml build requirements"
        )
        assert 'write_to = "f1pred/_version.py"' in content, (
            "setuptools-scm must write version to f1pred/_version.py"
        )

    def test_dockerfile_copies_git_dir(self):
        """Dockerfile must copy .git/ for setuptools-scm version derivation."""
        dockerfile = ROOT / "Dockerfile"
        assert dockerfile.exists(), "Dockerfile not found"
        content = dockerfile.read_text()
        assert ".git/" in content, (
            "Dockerfile must COPY .git/ so setuptools-scm can derive "
            "the version from git tags during the Docker build"
        )

    def test_prerelease_workflow_exists(self):
        """docker-publish.yml must build prerelease images on every push."""
        workflow = ROOT / ".github" / "workflows" / "docker-publish.yml"
        assert workflow.exists(), "docker-publish.yml not found"
        content = workflow.read_text()
        parsed = yaml.safe_load(content)
        triggers = parsed.get(True, {})  # 'on' parses as True in PyYAML
        assert "push" in triggers, (
            "docker-publish.yml must trigger on push for prerelease builds"
        )

    def test_prerelease_has_incrementing_version(self):
        """docker-publish.yml must produce numerically increasing prerelease tags."""
        workflow = ROOT / ".github" / "workflows" / "docker-publish.yml"
        content = workflow.read_text()
        assert "-pre." in content, (
            "docker-publish.yml must produce prerelease versions with "
            "'-pre.' suffix for Flux auto-pull compatibility"
        )

    def test_release_workflow_is_manual_only(self):
        """release.yml must only trigger via workflow_dispatch (manual)."""
        workflow = ROOT / ".github" / "workflows" / "release.yml"
        assert workflow.exists(), "release.yml not found"
        content = workflow.read_text()
        parsed = yaml.safe_load(content)
        triggers = parsed.get(True, {})
        assert "workflow_dispatch" in triggers, (
            "release.yml must support manual workflow_dispatch trigger"
        )
        assert "push" not in triggers, (
            "release.yml must NOT trigger on push — releases are manual decisions"
        )

    def test_release_workflow_builds_docker(self):
        """release.yml must build and push Docker images directly.

        Tags pushed by GITHUB_TOKEN do not trigger other workflows,
        so the release workflow must handle Docker builds itself.
        """
        workflow = ROOT / ".github" / "workflows" / "release.yml"
        content = workflow.read_text()
        assert "docker/build-push-action" in content, (
            "release.yml must include docker/build-push-action because "
            "tags pushed by GITHUB_TOKEN do not trigger other workflows"
        )
        assert "packages: write" in content, (
            "release.yml must have packages: write permission to push "
            "Docker images to GHCR"
        )

    def test_release_workflow_has_semver_tags(self):
        """release.yml must produce semver-tagged Docker images."""
        workflow = ROOT / ".github" / "workflows" / "release.yml"
        content = workflow.read_text()
        assert "type=semver" in content, (
            "release.yml must include type=semver tag patterns "
            "to produce versioned Docker images"
        )
