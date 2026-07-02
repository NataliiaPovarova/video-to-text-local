from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

from .errors import ProcessingError

_DEFAULT_SHIM_DIR = Path.home() / ".cache" / "video-to-text-local" / "bin"


def ensure_directories(paths: list[str], logger: logging.Logger) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)
        logger.debug("Ensured directory exists: %s", path)


def _shim_is_current(shim: Path, bundled: str) -> bool:
    """True when ``shim`` already exposes the current bundled binary.

    Handles both provisioning strategies: a symlink (compared by real path) and
    a plain copy (compared by size -- a cheap proxy that still catches an
    imageio-ffmpeg upgrade swapping in a differently-sized binary).
    """
    if shim.is_symlink():
        return os.path.realpath(shim) == os.path.realpath(bundled)
    if shim.is_file():
        try:
            return shim.stat().st_size == os.stat(bundled).st_size
        except OSError:
            return False
    return False


def _provision_shim(shim: Path, bundled: str) -> None:
    """Expose ``bundled`` at ``shim`` -- symlink when allowed, else copy.

    A symlink is cheap and self-updating, but ``os.symlink`` needs
    SeCreateSymbolicLinkPrivilege on Windows (admin or Developer Mode), which a
    normal user lacks. Fall back to copying the (self-contained, statically
    linked) ffmpeg binary so the bundled-ffmpeg path works with zero
    configuration on every platform. Any OSError from the copy propagates to the
    caller, which fails closed.
    """
    try:
        shim.symlink_to(bundled)
    except OSError:
        shutil.copy2(bundled, shim)
        os.chmod(shim, 0o755)


def ensure_ffmpeg_on_path(
    logger: logging.Logger | None = None,
    *,
    shim_dir: str | os.PathLike | None = None,
) -> str | None:
    """Guarantee a binary named ``ffmpeg`` is resolvable by name on PATH.

    openai-whisper's ``load_audio`` shells out to a bare ``ffmpeg`` (and the
    project defaults ``dependencies.ffmpeg_executable`` to ``"ffmpeg"`` too), so
    on a machine with no system ffmpeg, transcription dies instantly with
    ``FileNotFoundError: 'ffmpeg'``. moviepy already bundles a full ffmpeg via
    imageio-ffmpeg, so fall back to that binary -- exposing it under a
    PATH-resolvable name (``ffmpeg`` on POSIX, ``ffmpeg.exe`` on Windows so it
    satisfies PATHEXT) via a symlink, or a copy where symlinking is not
    permitted -- instead of forcing a separate system install.

    A real system ffmpeg always wins (newer/fuller build, no surprise PATH
    mutation). Returns the resolved ffmpeg path, or ``None`` when neither a
    system nor a bundled ffmpeg is available -- the caller then surfaces a clear
    "install ffmpeg" error via :func:`ensure_ffmpeg_available` rather than
    letting whisper fail deep in the pipeline. Idempotent.
    """
    log = logger or logging.getLogger("video_to_text")

    existing = shutil.which("ffmpeg")
    if existing:
        return existing

    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001 -- any failure means "no bundled ffmpeg"
        log.warning("No system ffmpeg and bundled imageio-ffmpeg unavailable: %s", exc)
        return None

    if not bundled or not os.path.exists(bundled):
        log.warning("No system ffmpeg and imageio-ffmpeg reported no usable binary.")
        return None

    shim_dir = Path(shim_dir) if shim_dir is not None else _DEFAULT_SHIM_DIR
    # On Windows, subprocess/shutil.which only resolve names whose extension is
    # in PATHEXT, so the shim MUST be ``ffmpeg.exe`` -- a bare ``ffmpeg`` would be
    # invisible to whisper's subprocess.run(["ffmpeg", ...]).
    shim = shim_dir / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
    try:
        shim_dir.mkdir(parents=True, exist_ok=True)
        # (Re)provision only when missing or stale (e.g. imageio-ffmpeg upgraded
        # its bundled binary).
        if not _shim_is_current(shim, bundled):
            if shim.is_symlink() or shim.exists():
                shim.unlink()
            _provision_shim(shim, bundled)
    except OSError as exc:
        log.warning("Could not provision ffmpeg shim at %s: %s", shim, exc)
        return None

    # Prepend the shim dir so a *bare* ``ffmpeg`` resolves to the bundled binary.
    # Guard against accumulating duplicate entries across repeated calls.
    path_entries = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
    if str(shim_dir) not in path_entries:
        os.environ["PATH"] = os.pathsep.join([str(shim_dir), *path_entries]) if path_entries else str(shim_dir)
    log.info("Using bundled ffmpeg via shim: %s -> %s", shim, bundled)
    return str(shim)


def ensure_ffmpeg_available(ffmpeg_executable: str, logger: logging.Logger) -> None:
    if shutil.which(ffmpeg_executable) is None:
        raise ProcessingError(
            f"'{ffmpeg_executable}' was not found on PATH. Whisper relies on ffmpeg to decode audio. "
            "Install ffmpeg and restart your terminal/IDE. "
            "On Windows with Chocolatey: choco install ffmpeg. "
            "Or download from https://ffmpeg.org/download.html and add its 'bin' folder to PATH."
        )
    logger.info("Dependency check passed: %s is available.", ffmpeg_executable)
