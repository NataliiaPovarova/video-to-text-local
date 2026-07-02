from src.utils.naming import build_output_basename, sanitize_stem


class TestSanitizeStem:
    def test_plain_name_unchanged(self):
        assert sanitize_stem("walkthrough") == "walkthrough"

    def test_spaces_become_underscores(self):
        assert sanitize_stem("my lecture file") == "my_lecture_file"

    def test_arabic_unicode_preserved_and_safe(self):
        out = sanitize_stem("محاضرة الفيزياء")
        assert out  # non-empty
        assert " " not in out
        # readable Arabic letters survive (the space between words is collapsed)
        assert "محاضرة" in out

    def test_illegal_characters_replaced(self):
        out = sanitize_stem('a/b\\c:d*e?f"g<h>i|j')
        for ch in '/\\:*?"<>|':
            assert ch not in out

    def test_overlong_truncated(self):
        out = sanitize_stem("x" * 500, max_len=80)
        assert len(out) <= 80

    def test_all_illegal_falls_back_to_id(self):
        assert sanitize_stem("///:::", fallback="job123") == "job123"

    def test_strips_leading_trailing_dots_and_spaces(self):
        assert sanitize_stem("  ..name..  ") == "name"

    def test_windows_reserved_device_name_prefixed(self):
        # Windows treats CON (with OR without an extension) as the CON device,
        # so the bare root name must be escaped with a leading "_".
        assert sanitize_stem("CON.mp4") == "_CON.mp4"
        assert sanitize_stem("con") == "_con"  # case-insensitive
        assert sanitize_stem("LPT9") == "_LPT9"

    def test_all_dots_with_extension_stays_non_empty(self):
        # A name that is only dots/extension must never sanitize to "" --
        # it falls back to the document id (here the default "transcript").
        out = sanitize_stem("....mp4")
        assert out  # non-empty


class TestBuildOutputBasename:
    def test_no_collision_returns_plain(self):
        out = build_output_basename("clip", "base", exists=lambda b: False, job_id="abcd1234ef")
        assert out == "clip__base"

    def test_collision_appends_job_suffix(self):
        out = build_output_basename("clip", "base", exists=lambda b: b == "clip__base", job_id="abcd1234ef")
        assert out == "clip__base__abcd1234"

    def test_different_models_distinct(self):
        a = build_output_basename("clip", "base", exists=lambda b: False, job_id="x")
        b = build_output_basename("clip", "large-v3", exists=lambda b: False, job_id="x")
        assert a != b
