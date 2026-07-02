from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

# Path separators, OS-reserved/illegal filename characters, and ASCII control
# chars. Covers the Windows-reserved set plus POSIX separators.
_ILLEGAL = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_WHITESPACE = re.compile(r"\s+")
_UNDERSCORES = re.compile(r"_+")

# Windows reserved device names. A file whose root name (the part before the
# first ".") matches one of these is unusable on Windows REGARDLESS of any
# extension -- "CON.mp4" is still the CON device. The user's platform is
# Windows, so guard these. Compared case-insensitively.
_RESERVED_NAMES = (
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


def sanitize_stem(name: str, *, fallback: str = "transcript", max_len: int = 80) -> str:
    """Normalize an uploaded file's stem into a filesystem-safe, readable stem.

    - NFC-normalizes Unicode (keeps readable Arabic/CJK where filesystem-safe)
    - replaces path separators / OS-reserved / control chars with "_"
    - collapses whitespace runs to "_" and repeated "_" to a single "_"
    - strips leading/trailing dots, spaces, and underscores
    - truncates to ``max_len`` characters
    - prefixes "_" when the root name is a Windows reserved device name
      (CON, PRN, AUX, NUL, COM1-9, LPT1-9), so the file is writable on Windows
    - returns ``fallback`` when nothing safe remains. Callers pass the document
      id as ``fallback`` (it defaults to "transcript" only when no id is given),
      so an all-dots / all-illegal name still yields a usable, unique stem.
    """
    stem = unicodedata.normalize("NFC", name)
    stem = _ILLEGAL.sub("_", stem)
    stem = _WHITESPACE.sub("_", stem)
    stem = _UNDERSCORES.sub("_", stem)
    stem = stem.strip(" ._")
    stem = stem[:max_len].strip(" ._")
    if not stem:
        return fallback
    # Reserved-name check runs on the sanitized stem so it can't be bypassed by
    # illegal chars; the "_" prefix makes "CON" -> "_CON", "CON.mp4" -> "_CON.mp4".
    root = stem.split(".", 1)[0]
    if root.upper() in _RESERVED_NAMES:
        stem = "_" + stem
    return stem


def build_output_basename(
    stem: str,
    model: str,
    *,
    exists: Callable[[str], bool],
    job_id: str,
) -> str:
    """Return a collision-safe output basename ``<stem>__<model>``.

    ``exists(basename)`` reports whether an output with that basename already
    exists on disk. On collision (same source + same model run again), append a
    short job-id suffix so multiple transcriptions coexist without overwriting.
    """
    base = f"{stem}__{model}"
    if not exists(base):
        return base
    return f"{stem}__{model}__{job_id[:8]}"
