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

    def test_tests_workflow_runs_on_workflow_call(self):
        """tests.yml must only run on workflow_call."""
        workflow = ROOT / ".github" / "workflows" / "tests.yml"
        assert workflow.exists(), "tests.yml not found"
        parsed = yaml.safe_load(workflow.read_text())
        triggers = parsed.get(True, {})  # 'on' parses as True in PyYAML
        assert "workflow_call" in triggers, (
            "tests.yml must trigger on workflow_call to run tests on every commit"
        )

    def test_build_workflow_triggers_on_push(self):
        """build.yml must build Docker images after tests pass."""
        workflow = ROOT / ".github" / "workflows" / "build.yml"
        assert workflow.exists(), "build.yml not found"
        content = workflow.read_text()
        parsed = yaml.safe_load(content)
        triggers = parsed.get(True, {})
        assert "push" in triggers, (
            "build.yml must trigger on push to build after tests pass"
        )

    def test_build_workflow_triggers_on_release(self):
        """build.yml must also trigger when a GitHub Release is published."""
        workflow = ROOT / ".github" / "workflows" / "build.yml"
        content = workflow.read_text()
        parsed = yaml.safe_load(content)
        triggers = parsed.get(True, {})
        assert "release" in triggers, (
            "build.yml must trigger on release events for stable builds"
        )
        release_types = triggers["release"].get("types", [])
        assert "published" in release_types, (
            "build.yml must trigger on release type 'published'"
        )

    def test_build_workflow_has_prerelease_versions(self):
        """build.yml must produce numerically increasing prerelease tags."""
        workflow = ROOT / ".github" / "workflows" / "build.yml"
        content = workflow.read_text()
        assert "-pre." in content, (
            "build.yml must produce prerelease versions with "
            "'-pre.' suffix for Flux auto-pull compatibility"
        )

    def test_build_workflow_has_semver_tags(self):
        """build.yml must produce semver-tagged Docker images for releases."""
        workflow = ROOT / ".github" / "workflows" / "build.yml"
        content = workflow.read_text()
        assert "type=semver" in content, (
            "build.yml must include type=semver tag patterns "
            "to produce versioned Docker images"
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

    def test_old_docker_publish_removed(self):
        """docker-publish.yml must not exist (replaced by build.yml)."""
        old_workflow = ROOT / ".github" / "workflows" / "docker-publish.yml"
        assert not old_workflow.exists(), (
            "docker-publish.yml should be deleted — "
            "Docker builds are now handled by build.yml"
        )
