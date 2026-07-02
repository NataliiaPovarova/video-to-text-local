from src.history import HistoryEntry, HistoryStore


def _entry(eid, name, outputs=None):
    return HistoryEntry(
        id=eid, source_filename=name, model="base", language="en",
        options={"cleanup": False, "diarize": False, "num_speakers": None},
        duration_seconds=12.3, word_count=4, created_at="2026-06-19T10:00:00+00:00",
        status="success", outputs=outputs or {},
    )


def test_add_and_list_newest_first(tmp_path):
    store = HistoryStore(tmp_path / ".history.json")
    store.add(_entry("a", "first.mp3"))
    store.add(_entry("b", "second.mp3"))
    rows = store.list()
    assert [r["id"] for r in rows] == ["b", "a"]


def test_get_returns_record(tmp_path):
    store = HistoryStore(tmp_path / ".history.json")
    store.add(_entry("a", "x.mp3"))
    assert store.get("a")["source_filename"] == "x.mp3"
    assert store.get("missing") is None


def test_delete_removes_record_and_files(tmp_path):
    out = tmp_path / "x__base.txt"
    out.write_text("hi", encoding="utf-8")
    store = HistoryStore(tmp_path / ".history.json")
    store.add(_entry("a", "x.mp3", outputs={"txt": str(out)}))
    assert store.delete("a") is True
    assert store.get("a") is None
    assert not out.exists()
    assert store.delete("a") is False


def test_search_by_filename_and_date(tmp_path):
    store = HistoryStore(tmp_path / ".history.json")
    store.add(_entry("a", "lecture.mp3"))
    store.add(_entry("b", "interview.mp4"))
    assert [r["id"] for r in store.search("lecture")] == ["a"]
    assert {r["id"] for r in store.search("2026-06-19")} == {"a", "b"}
    assert len(store.search("")) == 2


def test_survives_corrupt_index(tmp_path):
    path = tmp_path / ".history.json"
    path.write_text("{not json", encoding="utf-8")
    store = HistoryStore(path)
    assert store.list() == []
    store.add(_entry("a", "x.mp3"))
    assert store.get("a") is not None
