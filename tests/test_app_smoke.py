import pytest

# The GUI lives behind the optional requirements-gui.txt profile. Under a
# CLI-only install (requirements.txt, no gradio) these tests can't import `app`;
# skip them so the suite stays green for that profile instead of erroring.
pytest.importorskip("gradio")


def test_app_builds_blocks():
    import app

    demo = app.build_ui()
    # gradio Blocks exposes .launch; assert we constructed something launchable
    assert hasattr(demo, "launch")


def test_history_row_mapping_handles_empty():
    import app

    assert app._history_rows([]) == []


def test_monotonic_progress_drops_backward_updates():
    import app

    seen = []
    cb = app._make_monotonic_progress(lambda f, desc=None: seen.append(f))
    for f in (0.0, 0.5, 0.3, 1.0, 0.9):  # 0.3 and 0.9 are backward -> dropped
        cb(f)
    assert seen == [0.0, 0.5, 1.0]


def test_get_api_info_tolerates_boolean_schema_nodes():
    # Regression: gradio_client 1.3.0 crashes walking a boolean JSON-schema node
    # (`additionalProperties: true`, emitted by pydantic 2.13 for open dict
    # types) -> TypeError -> HTTP 500 on "/" -> gradio's launch-time url_ok sees
    # the 500 and aborts with "localhost is not accessible", so the GUI never
    # serves. Importing `app` installs a shim that treats non-dict schema nodes
    # as "Any". Lock that in at both the unit and the end-to-end level.
    import app
    import gradio_client.utils as gcu

    # A bare bool node and an object with boolean additionalProperties must not
    # raise and must yield a type-hint string.
    assert isinstance(gcu._json_schema_to_python_type(True, None), str)
    assert isinstance(
        gcu._json_schema_to_python_type(
            {"type": "object", "additionalProperties": True}, None
        ),
        str,
    )

    # End-to-end: building the app and computing its API info (what loading "/"
    # does) must complete instead of raising.
    demo = app.build_ui()
    info = demo.get_api_info()
    assert isinstance(info, dict)
    assert "named_endpoints" in info


def test_run_batch_continues_after_file_failure(monkeypatch):
    import gradio as gr
    import app

    called_paths = []
    FAIL_PATH = "b.mp3"
    GOOD_RETURN = ("t", "t.txt", None, None, gr.update(), gr.update())

    def fake_run_transcription(file_path, model, language, do_cleanup, do_diarize, num_speakers, progress=None):
        called_paths.append(file_path)
        if file_path == FAIL_PATH:
            raise gr.Error("boom")
        return GOOD_RETURN

    monkeypatch.setattr(app, "_run_transcription", fake_run_transcription)
    monkeypatch.setattr(app.gr, "Warning", lambda *a, **k: None)

    result = app._run_batch(
        ["a.mp3", "b.mp3", "c.mp3"],
        "base", "Auto-detect", False, False, None,
        progress=lambda *a, **k: None,
    )

    assert called_paths == ["a.mp3", "b.mp3", "c.mp3"], (
        "Expected _run_transcription called for all three files; got: " + str(called_paths)
    )
    assert isinstance(result, tuple) and len(result) == 6
