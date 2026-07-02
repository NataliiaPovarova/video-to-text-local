from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class HistoryEntry:
    id: str
    source_filename: str
    model: str
    language: str | None
    options: dict
    duration_seconds: float | None
    word_count: int
    created_at: str  # UTC ISO-8601
    status: str  # "success" | "failed" | "warning"
    outputs: dict  # {"txt":..., "clean":..., "srt":..., "vtt":...}
    message: str | None = None


class HistoryStore:
    """Atomic, lock-guarded JSON index of transcription jobs.

    Safe because jobs are serialized (queue concurrency 1). Newest first.
    """

    def __init__(self, path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    def _read(self) -> list[dict]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _write(self, records: list[dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.parent / (self._path.name + ".tmp")
        tmp.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)

    def add(self, entry: HistoryEntry) -> None:
        with self._lock:
            records = self._read()
            records.insert(0, asdict(entry))
            self._write(records)

    def list(self) -> list[dict]:
        with self._lock:
            return self._read()

    def get(self, entry_id: str) -> dict | None:
        return next((r for r in self.list() if r.get("id") == entry_id), None)

    def delete(self, entry_id: str) -> bool:
        with self._lock:
            records = self._read()
            removed = [r for r in records if r.get("id") == entry_id]
            if not removed:
                return False
            for r in removed:
                for p in (r.get("outputs") or {}).values():
                    if p:
                        try:
                            Path(p).unlink(missing_ok=True)
                        except OSError:
                            pass
            self._write([r for r in records if r.get("id") != entry_id])
            return True

    def search(self, query: str) -> list[dict]:
        q = (query or "").strip().lower()
        rows = self.list()
        if not q:
            return rows
        return [
            r for r in rows
            if q in r.get("source_filename", "").lower() or q in r.get("created_at", "").lower()
        ]
