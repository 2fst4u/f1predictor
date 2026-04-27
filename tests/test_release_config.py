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
        """build.yml must trigger on push to ensure every commit is gated."""
        workflow = ROOT / ".github" / "workflows" / "build.yml"
        assert workflow.exists(), "build.yml not found"
        content = workflow.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        triggers = parsed.get(True, {})
        assert "push" in triggers, (
            "build.yml must trigger on push to ensure every commit is gated and build Docker images after tests pass"
        )

    def test_build_workflow_triggers_on_pull_request(self):
        """build.yml must trigger on pull_request so the OpenCode review action
        (which does not support 'push' events) can run on every push to a PR
        branch via 'synchronize'."""
        workflow = ROOT / ".github" / "workflows" / "build.yml"
        content = workflow.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        triggers = parsed.get(True, {})
        assert "pull_request" in triggers, (
            "build.yml must trigger on pull_request so reviews run via the "
            "OpenCode action (which does not support push events)"
        )
        pr_types = triggers["pull_request"].get("types", []) or []
        for required in ("opened", "synchronize"):
            assert required in pr_types, (
                f"build.yml pull_request trigger must include '{required}' "
                f"to ensure reviews run on PR open and on every push to the PR branch"
            )

    def test_tests_job_skips_pull_request(self):
        """The tests job must skip on pull_request to avoid running tests
        twice (push and pull_request both fire on a PR-branch push)."""
        workflow = ROOT / ".github" / "workflows" / "build.yml"
        content = workflow.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        tests_job = parsed["jobs"]["tests"]
        tests_if = tests_job.get("if", "")
        # Assert exact behaviour rather than substrings
        assert tests_if == "github.event_name != 'pull_request'", (
            "tests job must exactly skip pull_request events"
        )

    def test_review_job_runs_on_pull_request(self):
        """The review job must be gated on pull_request events, not push.
        The OpenCode GitHub action exits with 'Unsupported event type: push'
        when invoked from a push context."""
        workflow = ROOT / ".github" / "workflows" / "build.yml"
        content = workflow.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        review_job = parsed["jobs"]["review"]
        review_if = review_job.get("if", "")
        # Assert exact behaviour rather than substrings
        assert review_if == "github.event_name == 'pull_request'", (
            "review job must exactly gate on pull_request events"
        )

    def test_build_job_dependencies(self):
        """The build job must depend ONLY on tests, not review, to ensure
        non-pull_request runs (like release or workflow_call) can execute
        without waiting for a PR-only review job."""
        workflow = ROOT / ".github" / "workflows" / "build.yml"
        content = workflow.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        build_job = parsed["jobs"]["build"]
        
        needs = build_job.get("needs", [])
        if isinstance(needs, str):
            needs = [needs]
            
        assert "tests" in needs, "build job must depend on tests"
        assert "review" not in needs, (
            "build job must NOT depend on review. Since review only runs on "
            "pull_request events and build only runs on push events, adding "
            "review to needs breaks the workflow on release/workflow_call events."
        )

    def test_build_job_skips_pull_request(self):
        """The build job must skip on pull_request events to avoid duplicate
        Docker builds (a push to a PR branch fires both push and pull_request)."""
        workflow = ROOT / ".github" / "workflows" / "build.yml"
        content = workflow.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        build_job = parsed["jobs"]["build"]
        build_if = build_job.get("if", "")
        
        # We need to verify that always() and != 'pull_request' is present
        # and that no other job result checks exist (especially needs.review)
        assert "always()" in build_if, "build job must run even if previous skipped"
        assert "github.event_name != 'pull_request'" in build_if, "build job must explicitly exclude pull_request events"
        assert "needs.tests.result == 'success'" in build_if, "build job must wait for tests success"
        assert "needs.review" not in build_if, "build job must NOT check review job status"

    def test_build_workflow_triggers_on_release(self):
        """build.yml must also trigger when a GitHub Release is published."""
        workflow = ROOT / ".github" / "workflows" / "build.yml"
        content = workflow.read_text(encoding="utf-8")
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
        content = workflow.read_text(encoding="utf-8")
        assert "-pre." in content, (
            "build.yml must produce prerelease versions with "
            "'-pre.' suffix for Flux auto-pull compatibility"
        )

    def test_build_workflow_has_semver_tags(self):
        """build.yml must produce semver-tagged Docker images for releases."""
        workflow = ROOT / ".github" / "workflows" / "build.yml"
        content = workflow.read_text(encoding="utf-8")
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
