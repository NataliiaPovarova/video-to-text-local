from __future__ import annotations

import os
from pathlib import Path


def discover_media_files(
    folder_path: str | Path,
    supported_extensions: tuple[str, ...],
) -> list[tuple[str, Path]]:
    """Scan a directory for media files with matching extensions.

    Returns a list of (filename, full_path) tuples.
    """
    folder = Path(folder_path)
    files: list[tuple[str, Path]] = []
    if not folder.is_dir():
        return files

    for entry in sorted(folder.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() in supported_extensions:
            files.append((entry.name, entry))

    return files
