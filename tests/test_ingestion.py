from pathlib import Path

from src.ingestion.file_loader import discover_media_files


class TestDiscoverMediaFiles:
    def test_empty_directory(self, tmp_path: Path):
        result = discover_media_files(tmp_path, (".mp3", ".wav"))
        assert result == []

    def test_finds_matching_files(self, tmp_path: Path):
        (tmp_path / "song.mp3").write_text("fake audio")
        (tmp_path / "video.mp4").write_text("fake video")
        (tmp_path / "readme.txt").write_text("not media")

        result = discover_media_files(tmp_path, (".mp3", ".mp4"))
        names = [name for name, _ in result]
        assert "song.mp3" in names
        assert "video.mp4" in names
        assert "readme.txt" not in names

    def test_case_insensitive_extension(self, tmp_path: Path):
        (tmp_path / "LOUD.MP3").write_text("fake")
        result = discover_media_files(tmp_path, (".mp3",))
        assert len(result) == 1

    def test_skips_directories(self, tmp_path: Path):
        (tmp_path / "subdir.mp3").mkdir()
        result = discover_media_files(tmp_path, (".mp3",))
        assert result == []

    def test_nonexistent_directory(self):
        result = discover_media_files(Path("/nonexistent/path"), (".mp3",))
        assert result == []

    def test_sorted_output(self, tmp_path: Path):
        (tmp_path / "c.mp3").write_text("")
        (tmp_path / "a.mp3").write_text("")
        (tmp_path / "b.mp3").write_text("")
        result = discover_media_files(tmp_path, (".mp3",))
        names = [name for name, _ in result]
        assert names == ["a.mp3", "b.mp3", "c.mp3"]
