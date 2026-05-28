"""Weight resolution and download for vessel segmentation checkpoints from fundus-image-toolbox."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Union
from urllib.error import URLError
from urllib.request import Request, urlopen

ENSEMBLE_WEIGHT_FILES = tuple(f"FRUNet_{i}.pth" for i in range(5))
APP_CACHE_DIRNAME = "segmentation_quality_control"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 300

VESSEL_WEIGHTS_GIT_URL = (
    "https://github.com/berenslab/MIDL24-segmentation_quality_control.git"
)
# Weights location on GitHub; PyPI wheels do not bundle checkpoint files.
VESSEL_WEIGHTS_GIT_REF = "make-package-without-bunch"
VESSEL_WEIGHTS_GIT_SUBDIR = "trained"
VESSEL_WEIGHTS_RAW_BASE = (
    "https://raw.githubusercontent.com/berenslab/MIDL24-segmentation_quality_control"
    f"/{VESSEL_WEIGHTS_GIT_REF}/{VESSEL_WEIGHTS_GIT_SUBDIR}"
)


def has_all_paths(
    base_dir: Union[str, Path], relative_paths: Iterable[Union[str, Path]]
) -> bool:
    """Check if all expected paths exist under base_dir."""
    base = Path(base_dir).expanduser().resolve()
    return all((base / Path(p)).exists() for p in relative_paths)


def _remove_file_if_exists(path: Union[str, Path]) -> None:
    p = Path(path)
    try:
        if p.exists() and p.is_file():
            os.remove(str(p))
    except OSError:
        pass


def _format_megabytes(num_bytes: int) -> str:
    return f"{num_bytes / 1_000_000:.1f} MB"


def _print_download_progress(name: str, downloaded: int, total: Optional[int]) -> None:
    prefix = f"[segmentation_quality_control] {name}: "
    if total:
        pct = min(100, 100 * downloaded / total)
        line = (
            f"{prefix}{_format_megabytes(downloaded)} / "
            f"{_format_megabytes(total)} ({pct:5.1f}%)"
        )
    else:
        line = f"{prefix}{_format_megabytes(downloaded)} downloaded"
    print(f"\r{line}", end="", flush=True)


def resolve_package_dir() -> Path:
    """Return the installed ``segmentation_quality_control`` package directory."""
    return Path(__file__).resolve().parents[2]


def _local_trained_dir_candidates() -> tuple[Path, ...]:
    """Candidate ``trained/`` locations for editable or source-tree installs."""
    package_dir = resolve_package_dir()
    return (
        package_dir.parent / VESSEL_WEIGHTS_GIT_SUBDIR,
        package_dir / VESSEL_WEIGHTS_GIT_SUBDIR,
    )


def resolve_local_trained_dir() -> Optional[Path]:
    """Return a nearby ``trained/`` directory when all checkpoints are present."""
    for candidate in _local_trained_dir_candidates():
        if _has_ensemble_weights(candidate):
            return candidate.resolve()
    return None


def resolve_cache_dir(cache_dir: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the user cache root for downloaded checkpoints."""
    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve()

    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA")
        if root:
            return Path(root).resolve() / APP_CACHE_DIRNAME
        return (Path.home() / "AppData" / "Local" / APP_CACHE_DIRNAME).resolve()

    if os.name == "posix" and Path.home().exists():
        if os.uname().sysname == "Darwin":
            return (
                Path.home() / "Library" / "Caches" / APP_CACHE_DIRNAME
            ).resolve()
        return (Path.home() / ".cache" / APP_CACHE_DIRNAME).resolve()

    return (Path.cwd() / ".cache" / APP_CACHE_DIRNAME).resolve()


def resolve_cache_models_dir(
    cache_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Return the cache directory used for downloaded ensemble checkpoints."""
    trained_dir = resolve_cache_dir(cache_dir=cache_dir) / VESSEL_WEIGHTS_GIT_SUBDIR
    trained_dir.mkdir(parents=True, exist_ok=True)
    return trained_dir


def _has_ensemble_weights(models_dir: Path) -> bool:
    return has_all_paths(models_dir, ENSEMBLE_WEIGHT_FILES)


def resolve_models_dir(
    cache_dir: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve the directory containing FR-UNet ensemble checkpoints."""
    if models_dir is not None:
        resolved = Path(models_dir).expanduser().resolve()
        if not _has_ensemble_weights(resolved):
            missing = [
                name
                for name in ENSEMBLE_WEIGHT_FILES
                if not (resolved / name).exists()
            ]
            raise FileNotFoundError(
                "[segmentation_quality_control] Expected weights missing in "
                f"{resolved}: {', '.join(missing)}"
            )
        return resolved

    local_trained = resolve_local_trained_dir()
    if local_trained is not None:
        return local_trained

    return resolve_cache_models_dir(cache_dir=cache_dir)


def _manual_download_instructions(models_dir: Path) -> str:
    wget_lines = "\n".join(
        f"  wget -O {models_dir / name} {VESSEL_WEIGHTS_RAW_BASE}/{name}"
        for name in ENSEMBLE_WEIGHT_FILES
    )
    return (
        "Manual workaround to get the ensemble weights:\n"
        "Option A — direct download (wget or curl):\n"
        f"  mkdir -p {models_dir}\n"
        f"{wget_lines}\n"
        "  # or, with curl:\n"
        + "\n".join(
            f"  curl -L -o {models_dir / name} {VESSEL_WEIGHTS_RAW_BASE}/{name}"
            for name in ENSEMBLE_WEIGHT_FILES
        )
        + "\n"
        "Option B — sparse git checkout:\n"
        f"1) git clone --depth 1 --filter=blob:none --sparse "
        f"{VESSEL_WEIGHTS_GIT_URL} ~/tmp/segmentation_weights\n"
        f"2) cd ~/tmp/segmentation_weights\n"
        f"3) git sparse-checkout set {VESSEL_WEIGHTS_GIT_SUBDIR}\n"
        f"4) git checkout {VESSEL_WEIGHTS_GIT_REF}\n"
        f"5) Copy {', '.join(ENSEMBLE_WEIGHT_FILES)} into {models_dir}\n"
        f"6) Re-run your command."
    )


def _run_git(args: list[str], *, cwd: Optional[Path] = None) -> None:
    result = subprocess.run(
        args,
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or f"exit code {result.returncode}"
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")


def _download_checkpoint_file(url: str, target: Path) -> None:
    """Download one checkpoint with a simple terminal progress line."""
    request = Request(url, headers={"User-Agent": "segmentation_quality_control"})
    partial = target.with_suffix(target.suffix + ".part")
    downloaded = 0

    try:
        with urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header else None
            with partial.open("wb") as handle:
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    _print_download_progress(target.name, downloaded, total)
        print(file=sys.stderr)
        partial.replace(target)
    except Exception:
        _remove_file_if_exists(partial)
        raise


def _fetch_weights_from_http(
    destination_dir: Path,
    raw_base: str = VESSEL_WEIGHTS_RAW_BASE,
) -> None:
    """Download ensemble checkpoints over HTTPS into destination_dir."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    try:
        for name in ENSEMBLE_WEIGHT_FILES:
            target = (destination_dir / name).resolve()
            if not str(target).startswith(str(destination_dir.resolve())):
                raise RuntimeError(
                    "[segmentation_quality_control] Unsafe destination path "
                    f"for {name}"
                )

            url = f"{raw_base}/{name}"
            print(
                "[segmentation_quality_control] Downloading "
                f"{name} from GitHub..."
            )
            _download_checkpoint_file(url, target)
            downloaded.append(target)
    except Exception:
        for path in downloaded:
            _remove_file_if_exists(path)
        raise


def _fetch_weights_from_git(
    destination_dir: Path,
    git_url: str = VESSEL_WEIGHTS_GIT_URL,
    git_ref: str = VESSEL_WEIGHTS_GIT_REF,
    subdir: str = VESSEL_WEIGHTS_GIT_SUBDIR,
) -> None:
    """Fetch ensemble checkpoints from git into destination_dir."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []

    try:
        with tempfile.TemporaryDirectory(prefix="segqc-weights-") as tmp:
            repo_dir = Path(tmp) / "repo"
            _run_git(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--filter=blob:none",
                    "--sparse",
                    "--no-checkout",
                    git_url,
                    str(repo_dir),
                ]
            )
            _run_git(["git", "sparse-checkout", "set", subdir], cwd=repo_dir)
            _run_git(
                ["git", "fetch", "--depth", "1", "origin", git_ref],
                cwd=repo_dir,
            )
            _run_git(["git", "checkout", "FETCH_HEAD"], cwd=repo_dir)

            source_dir = (repo_dir / subdir).resolve()
            if not source_dir.is_dir():
                raise FileNotFoundError(
                    "[segmentation_quality_control] Missing "
                    f"'{subdir}/' directory at git ref {git_ref} in {git_url}"
                )

            for name in ENSEMBLE_WEIGHT_FILES:
                source = (source_dir / name).resolve()
                if not source.is_file():
                    raise FileNotFoundError(
                        "[segmentation_quality_control] Missing checkpoint "
                        f"{name} at git ref {git_ref} in {git_url}"
                    )
                if not str(source).startswith(str(source_dir)):
                    raise RuntimeError(
                        "[segmentation_quality_control] Unsafe checkpoint path "
                        f"for {name}"
                    )

                target = (destination_dir / name).resolve()
                if not str(target).startswith(str(destination_dir.resolve())):
                    raise RuntimeError(
                        "[segmentation_quality_control] Unsafe destination path "
                        f"for {name}"
                    )

                shutil.copy2(source, target)
                copied.append(target)
    except Exception:
        for path in copied:
            _remove_file_if_exists(path)
        raise


def _fetch_weights(
    destination_dir: Path,
    raw_base: str = VESSEL_WEIGHTS_RAW_BASE,
    git_url: str = VESSEL_WEIGHTS_GIT_URL,
    git_ref: str = VESSEL_WEIGHTS_GIT_REF,
    subdir: str = VESSEL_WEIGHTS_GIT_SUBDIR,
) -> None:
    """Download checkpoints via HTTPS, falling back to git sparse checkout."""
    try:
        _fetch_weights_from_http(destination_dir, raw_base=raw_base)
    except (OSError, RuntimeError, URLError, TimeoutError, ValueError) as http_exc:
        if shutil.which("git") is None:
            raise RuntimeError(
                "[segmentation_quality_control] HTTPS download failed and git "
                f"is not available on PATH. Last error: {http_exc}"
            ) from http_exc

        print(
            "[segmentation_quality_control] HTTPS download failed; "
            f"trying git sparse checkout ({git_ref}:{subdir}/)...",
            file=sys.stderr,
        )
        _fetch_weights_from_git(
            destination_dir,
            git_url=git_url,
            git_ref=git_ref,
            subdir=subdir,
        )


def download_weights(
    raw_base: str = VESSEL_WEIGHTS_RAW_BASE,
    git_url: str = VESSEL_WEIGHTS_GIT_URL,
    git_ref: str = VESSEL_WEIGHTS_GIT_REF,
    cache_dir: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Download ensemble weights into ``models_dir`` or the default cache."""
    if models_dir is not None:
        destination = Path(models_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
    else:
        destination = resolve_models_dir(cache_dir=cache_dir)

    if _has_ensemble_weights(destination):
        return destination

    print(
        "[segmentation_quality_control] Downloading ensemble weights "
        f"({raw_base}/)..."
    )
    try:
        _fetch_weights(
            destination_dir=destination,
            raw_base=raw_base,
            git_url=git_url,
            git_ref=git_ref,
        )
        print("[segmentation_quality_control] Done.")
    except Exception as exc:
        raise RuntimeError(
            "[segmentation_quality_control] Failed to download weights.\n"
            f"Last error: {exc}\n\n{_manual_download_instructions(destination)}"
        ) from exc

    if not _has_ensemble_weights(destination):
        raise FileNotFoundError(
            "[segmentation_quality_control] Expected checkpoints were not found "
            f"after download in {destination}: "
            f"{', '.join(ENSEMBLE_WEIGHT_FILES)}"
        )
    return destination


def ensure_models_dir(
    cache_dir: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Return a models directory, downloading weights if needed."""
    if models_dir is not None:
        target = Path(models_dir).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        if _has_ensemble_weights(target):
            return target
        return download_weights(models_dir=target)

    resolved = resolve_models_dir(cache_dir=cache_dir)
    if _has_ensemble_weights(resolved):
        return resolved

    return download_weights(cache_dir=cache_dir)
