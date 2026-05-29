"""Tests for checkpoint weight resolution and download."""

from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

import pytest

from segmentation_quality_control.utils.checkpoint.weights import (
    ENSEMBLE_WEIGHT_FILES,
    VESSEL_WEIGHTS_GIT_REF,
    VESSEL_WEIGHTS_GIT_SUBDIR,
    VESSEL_WEIGHTS_GIT_URL,
    VESSEL_WEIGHTS_RAW_BASE,
    _github_repo_slug,
    _fetch_weights,
    _has_ensemble_weights,
    _manual_download_instructions,
    download_weights,
    ensure_models_dir,
    resolve_cache_models_dir,
    resolve_local_trained_dir,
    resolve_models_dir,
)


def write_dummy_ensemble(trained_dir: Path) -> Path:
    """Create placeholder checkpoint files for resolution tests."""
    trained_dir.mkdir(parents=True, exist_ok=True)
    for name in ENSEMBLE_WEIGHT_FILES:
        (trained_dir / name).write_bytes(b"dummy")
    return trained_dir


def mock_dev_tree(tmp_path: Path) -> tuple[Path, Path]:
    """Return (package_dir, repo_trained_dir) for a source-tree layout."""
    repo_root = tmp_path / "project"
    package_dir = repo_root / "segmentation_quality_control"
    package_dir.mkdir(parents=True)
    repo_trained = write_dummy_ensemble(repo_root / "trained")
    return package_dir, repo_trained


class TestResolveModelsDir:
    def test_explicit_models_dir(self, tmp_path):
        trained = write_dummy_ensemble(tmp_path / "custom")
        resolved = resolve_models_dir(models_dir=trained)
        assert resolved == trained.resolve()

    def test_explicit_models_dir_missing_raises(self, tmp_path):
        missing_dir = tmp_path / "empty"
        missing_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="Expected weights missing"):
            resolve_models_dir(models_dir=missing_dir)

    def test_ensure_models_dir_downloads_to_explicit_models_dir(self, tmp_path):
        target = tmp_path / "custom" / "weights"

        def fake_fetch(destination_dir: Path, **kwargs):
            write_dummy_ensemble(destination_dir)

        with patch(
            "segmentation_quality_control.utils.checkpoint.weights._fetch_weights_from_http",
            side_effect=fake_fetch,
        ) as fetch:
            resolved = ensure_models_dir(models_dir=target)

        assert resolved == target.resolve()
        fetch.assert_called_once()
        assert _has_ensemble_weights(target)

    def test_dev_local_trained_preferred_over_cache(self, tmp_path, monkeypatch):
        package_dir, repo_trained = mock_dev_tree(tmp_path)
        cache_root = tmp_path / "cache"

        monkeypatch.setattr(
            "segmentation_quality_control.utils.checkpoint.weights.resolve_package_dir",
            lambda: package_dir,
        )

        resolved = resolve_models_dir(cache_dir=cache_root)
        assert resolved == repo_trained.resolve()
        assert resolved != resolve_cache_models_dir(cache_dir=cache_root)

    def test_cache_used_when_no_local_trained(self, tmp_path, monkeypatch):
        package_dir = tmp_path / "project" / "segmentation_quality_control"
        package_dir.mkdir(parents=True)
        cache_root = tmp_path / "cache"

        monkeypatch.setattr(
            "segmentation_quality_control.utils.checkpoint.weights.resolve_package_dir",
            lambda: package_dir,
        )

        resolved = resolve_models_dir(cache_dir=cache_root)
        assert resolved == (cache_root / "trained").resolve()
        assert resolve_local_trained_dir() is None


class TestEnsureModelsDir:
    def test_skips_download_when_local_trained_present(self, tmp_path, monkeypatch):
        package_dir, repo_trained = mock_dev_tree(tmp_path)
        monkeypatch.setattr(
            "segmentation_quality_control.utils.checkpoint.weights.resolve_package_dir",
            lambda: package_dir,
        )

        with patch(
            "segmentation_quality_control.utils.checkpoint.weights._fetch_weights"
        ) as fetch:
            resolved = ensure_models_dir(cache_dir=tmp_path / "cache")

        assert resolved == repo_trained.resolve()
        fetch.assert_not_called()

    def test_downloads_to_cache_when_local_absent(self, tmp_path, monkeypatch):
        package_dir = tmp_path / "project" / "segmentation_quality_control"
        package_dir.mkdir(parents=True)
        cache_root = tmp_path / "cache"
        cache_trained = cache_root / "trained"

        monkeypatch.setattr(
            "segmentation_quality_control.utils.checkpoint.weights.resolve_package_dir",
            lambda: package_dir,
        )

        def fake_fetch(destination_dir: Path, **kwargs):
            write_dummy_ensemble(destination_dir)

        with patch(
            "segmentation_quality_control.utils.checkpoint.weights._fetch_weights_from_http",
            side_effect=fake_fetch,
        ) as fetch:
            resolved = ensure_models_dir(cache_dir=cache_root)

        assert resolved == cache_trained.resolve()
        fetch.assert_called_once()
        assert all((cache_trained / name).is_file() for name in ENSEMBLE_WEIGHT_FILES)

    def test_second_call_skips_download(self, tmp_path, monkeypatch):
        package_dir = tmp_path / "project" / "segmentation_quality_control"
        package_dir.mkdir(parents=True)
        cache_root = tmp_path / "cache"

        monkeypatch.setattr(
            "segmentation_quality_control.utils.checkpoint.weights.resolve_package_dir",
            lambda: package_dir,
        )

        def fake_fetch(destination_dir: Path, **kwargs):
            write_dummy_ensemble(destination_dir)

        with patch(
            "segmentation_quality_control.utils.checkpoint.weights._fetch_weights_from_http",
            side_effect=fake_fetch,
        ) as fetch:
            first = ensure_models_dir(cache_dir=cache_root)
            second = ensure_models_dir(cache_dir=cache_root)

        assert first == second
        fetch.assert_called_once()


class TestManualInstructions:
    def test_includes_wget_and_curl_urls(self, tmp_path):
        models_dir = tmp_path / "trained"
        text = _manual_download_instructions(models_dir)

        for name in ENSEMBLE_WEIGHT_FILES:
            assert f"{VESSEL_WEIGHTS_RAW_BASE}/{name}" in text
            assert f"wget -O {models_dir / name}" in text
            assert f"curl -L -o {models_dir / name}" in text

    def test_raw_base_matches_git_url(self):
        slug = _github_repo_slug(VESSEL_WEIGHTS_GIT_URL)
        assert VESSEL_WEIGHTS_RAW_BASE == (
            f"https://raw.githubusercontent.com/{slug}/"
            f"{VESSEL_WEIGHTS_GIT_REF}/{VESSEL_WEIGHTS_GIT_SUBDIR}"
        )


class TestFetchWeightsFallback:
    def test_uses_http_without_git_when_http_succeeds(self, tmp_path):
        destination = tmp_path / "trained"

        def fake_http(dest: Path, **kwargs):
            write_dummy_ensemble(dest)

        with patch(
            "segmentation_quality_control.utils.checkpoint.weights._fetch_weights_from_http",
            side_effect=fake_http,
        ) as http_fetch, patch(
            "segmentation_quality_control.utils.checkpoint.weights._fetch_weights_from_git"
        ) as git_fetch:
            _fetch_weights(destination)

        http_fetch.assert_called_once()
        git_fetch.assert_not_called()
        assert _has_ensemble_weights(destination)

    def test_falls_back_to_git_when_http_fails(self, tmp_path):
        destination = tmp_path / "trained"

        def fake_git(dest: Path, **kwargs):
            write_dummy_ensemble(dest)

        with patch(
            "segmentation_quality_control.utils.checkpoint.weights._fetch_weights_from_http",
            side_effect=URLError("network down"),
        ) as http_fetch, patch(
            "segmentation_quality_control.utils.checkpoint.weights._fetch_weights_from_git",
            side_effect=fake_git,
        ) as git_fetch, patch(
            "segmentation_quality_control.utils.checkpoint.weights.shutil.which",
            return_value="/usr/bin/git",
        ):
            _fetch_weights(destination)

        http_fetch.assert_called_once()
        git_fetch.assert_called_once()
        assert _has_ensemble_weights(destination)

    def test_raises_when_http_fails_and_git_missing(self, tmp_path):
        destination = tmp_path / "trained"

        with patch(
            "segmentation_quality_control.utils.checkpoint.weights._fetch_weights_from_http",
            side_effect=URLError("network down"),
        ), patch(
            "segmentation_quality_control.utils.checkpoint.weights._fetch_weights_from_git"
        ) as git_fetch, patch(
            "segmentation_quality_control.utils.checkpoint.weights.shutil.which",
            return_value=None,
        ):
            with pytest.raises(RuntimeError, match="git is not available on PATH"):
                _fetch_weights(destination)

        git_fetch.assert_not_called()


class TestUtilsExports:
    def test_utils_reexports_download_helpers(self):
        from segmentation_quality_control.utils import (
            ENSEMBLE_WEIGHT_FILES as utils_files,
            download_weights as utils_download,
            ensure_models_dir as utils_ensure,
            resolve_models_dir as utils_resolve,
        )

        assert utils_files is ENSEMBLE_WEIGHT_FILES
        assert utils_download is download_weights
        assert utils_ensure is ensure_models_dir
        assert utils_resolve is resolve_models_dir


class TestRepoDevInstall:
    """Uses the real checkout when repo ``trained/`` exists."""

    def test_repo_trained_is_discovered_in_editable_install(self):
        repo_trained = Path(__file__).resolve().parents[1] / "trained"
        if not all((repo_trained / name).exists() for name in ENSEMBLE_WEIGHT_FILES):
            pytest.skip("repo trained/ checkpoints not present")

        local = resolve_local_trained_dir()
        resolved = resolve_models_dir()
        assert local == repo_trained.resolve()
        assert resolved == repo_trained.resolve()


@pytest.mark.integration
def test_real_http_download_into_cache(tmp_path, monkeypatch):
    """PyPI-like path: no local trained/, download into an isolated cache dir."""
    package_dir = tmp_path / "isolated" / "segmentation_quality_control"
    package_dir.mkdir(parents=True)
    cache_root = tmp_path / "cache"

    monkeypatch.setattr(
        "segmentation_quality_control.utils.checkpoint.weights.resolve_package_dir",
        lambda: package_dir,
    )

    models_dir = ensure_models_dir(cache_dir=cache_root)
    assert models_dir == (cache_root / "trained").resolve()
    assert all((models_dir / name).is_file() for name in ENSEMBLE_WEIGHT_FILES)
    assert all((models_dir / name).stat().st_size > 1_000_000 for name in ENSEMBLE_WEIGHT_FILES)
