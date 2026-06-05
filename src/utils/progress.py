from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar

from tqdm import tqdm

T = TypeVar("T")


@contextmanager
def duration_progress_bar(
    *,
    desc: str,
    duration_seconds: float | None,
    update_interval_seconds: float = 0.25,
):
    """Show elapsed-time progress capped at audio duration (fallback when no native hook)."""
    if duration_seconds is None or duration_seconds <= 0:
        yield
        return

    stop_event = threading.Event()
    pbar: tqdm | None = None

    def _run() -> None:
        nonlocal pbar
        pbar = tqdm(total=int(duration_seconds), desc=desc, unit="s")
        start = time.time()
        while not stop_event.is_set():
            elapsed = time.time() - start
            current = int(min(elapsed, pbar.total))
            if current != pbar.n:
                pbar.n = current
                pbar.refresh()
            time.sleep(update_interval_seconds)
        pbar.n = pbar.total
        pbar.refresh()
        pbar.close()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=0.5)


def tqdm_iter(
    iterable: Iterator[T] | list[T],
    *,
    desc: str,
    unit: str = "it",
    total: int | None = None,
) -> Iterator[T]:
    """Iterate with a tqdm bar (used for short post-processing steps)."""
    yield from tqdm(iterable, desc=desc, unit=unit, total=total, leave=False)
