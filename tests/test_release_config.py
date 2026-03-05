"""Tests to validate that release infrastructure files stay consistent.

These are pure file-parsing tests (no network, no Docker) that catch accidental
breakage of the version/release/Docker toolchain.
"""
import os
from pathlib import Path

import pytest

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

    def test_docker_publish_has_semver_tags(self):
        """docker-publish.yml must have semver tag patterns for versioned images."""
        workflow = ROOT / ".github" / "workflows" / "docker-publish.yml"
        assert workflow.exists(), "docker-publish.yml not found"
        content = workflow.read_text()
        assert "type=semver" in content, (
            "docker-publish.yml must include type=semver tag patterns "
            "to produce versioned Docker images from git tags"
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

    def test_release_workflow_exists(self):
        """release.yml must exist with push and workflow_dispatch triggers."""
        workflow = ROOT / ".github" / "workflows" / "release.yml"
        assert workflow.exists(), "release.yml not found"
        content = workflow.read_text()
        assert "workflow_dispatch" in content, (
            "release.yml must support manual workflow_dispatch trigger"
        )
        assert "push:" in content, (
            "release.yml must trigger on push to main for auto-patch"
        )

    def test_docker_publish_triggers_on_tags(self):
        """docker-publish.yml must trigger on v*.*.* tag pushes."""
        workflow = ROOT / ".github" / "workflows" / "docker-publish.yml"
        content = workflow.read_text()
        assert "v*.*.*" in content, (
            "docker-publish.yml must trigger on v*.*.* tags so that "
            "release tags produce Docker images"
        )
