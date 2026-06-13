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

    def test_build_workflow_triggers_on_release(self):
        """build.yml must trigger on a published GitHub Release so that
        manually publishing a release (e.g. from the 'Draft a new release' UI)
        builds the stable container image."""
        workflow = ROOT / ".github" / "workflows" / "build.yml"
        content = workflow.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        triggers = parsed.get(True, {})
        assert "release" in triggers, (
            "build.yml must trigger on the 'release' event so a "
            "manually published GitHub Release builds the stable image"
        )
        types = (triggers.get("release") or {}).get("types", [])
        assert "published" in types, (
            "build.yml's release trigger must fire on the 'published' type"
        )

    def test_build_workflow_has_no_review_job(self):
        """build.yml must not define a review job. Code review is invoked
        manually via /oc-review (handled by opencode-review.yml), not as
        part of the CI pipeline."""
        workflow = ROOT / ".github" / "workflows" / "build.yml"
        content = workflow.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        jobs = parsed.get("jobs", {})
        assert "review" not in jobs, (
            "build.yml must not contain a 'review' job — reviews are "
            "manual via /oc-review"
        )

    def test_build_workflow_does_not_trigger_on_pull_request(self):
        """build.yml must NOT trigger on pull_request. The CI pipeline only
        runs tests and builds Docker images on push/release/workflow_call.
        Pull-request reviews are manual via /oc-review."""
        workflow = ROOT / ".github" / "workflows" / "build.yml"
        content = workflow.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        triggers = parsed.get(True, {})
        assert "pull_request" not in triggers, (
            "build.yml must not trigger on pull_request — reviews are "
            "manual via /oc-review and CI only runs on push/release"
        )

    def test_manual_review_workflow_exists(self):
        """opencode-review.yml must exist and respond to /oc-review
        comments on PRs/issues so reviews can be requested manually."""
        workflow = ROOT / ".github" / "workflows" / "opencode-review.yml"
        assert workflow.exists(), (
            "opencode-review.yml must exist to handle manual /oc-review "
            "requests"
        )
        content = workflow.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        triggers = parsed.get(True, {})
        # Must trigger on comment events (issue_comment for PR/issue
        # threads, pull_request_review_comment for inline code comments)
        assert "issue_comment" in triggers or "pull_request_review_comment" in triggers, (
            "opencode-review.yml must trigger on comment events to "
            "handle /oc-review slash commands"
        )
        assert "/oc-review" in content, (
            "opencode-review.yml must filter comments for the "
            "/oc-review slash command"
        )
        assert "anomalyco/opencode/github" in content, (
            "opencode-review.yml must invoke the OpenCode GitHub action"
        )

    def test_no_automatic_review_workflow(self):
        """There must be no workflow that runs the OpenCode review action
        automatically on push/pull_request. Reviews are strictly manual
        via /oc-review to keep CI fast and inexpensive."""
        pr_review = ROOT / ".github" / "workflows" / "pr-review.yml"
        assert not pr_review.exists(), (
            "pr-review.yml must not exist — automatic reviews were "
            "removed in favour of manual /oc-review only"
        )
        ci_fix = ROOT / ".github" / "workflows" / "opencode-ci-fix.yml"
        assert not ci_fix.exists(), (
            "opencode-ci-fix.yml must not exist — automatic /oc-review "
            "posting on CI failure was removed in favour of fully manual "
            "review invocation"
        )

    def test_build_workflow_supports_release_via_workflow_call(self):
        """Stable release builds are produced by invoking build.yml through
        workflow_call (from release.yml) with is_release=true, rather than via
        a direct release trigger on build.yml itself."""
        workflow = ROOT / ".github" / "workflows" / "build.yml"
        content = workflow.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        triggers = parsed.get(True, {})
        assert "workflow_call" in triggers, (
            "build.yml must be callable via workflow_call so release.yml can "
            "trigger stable release builds"
        )
        inputs = (triggers.get("workflow_call") or {}).get("inputs", {})
        assert "is_release" in inputs, (
            "build.yml workflow_call must accept an 'is_release' input to "
            "distinguish stable release builds from dev builds"
        )

        # release.yml must actually wire the stable build through build.yml.
        release = ROOT / ".github" / "workflows" / "release.yml"
        rel_content = release.read_text(encoding="utf-8")
        assert "./.github/workflows/build.yml" in rel_content, (
            "release.yml must call build.yml to produce the release image"
        )
        assert "is_release: true" in rel_content, (
            "release.yml must invoke build.yml with is_release: true"
        )

    def test_build_workflow_has_prerelease_versions(self):
        """build.yml must produce numerically increasing dev prerelease tags
        (the '-dev.N' scheme) for non-release pushes."""
        workflow = ROOT / ".github" / "workflows" / "build.yml"
        content = workflow.read_text(encoding="utf-8")
        assert "-dev." in content, (
            "build.yml must produce dev prerelease versions with the "
            "'-dev.' suffix for non-release pushes"
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
