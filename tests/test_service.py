from pathlib import Path

import src.service as service
from src.models import PipelineState, TranscriptDocument, TranscriptSegment


def _stub_doc():
    return TranscriptDocument(
        segments=[
            TranscriptSegment(text="Hello world.", start_time=0.0, end_time=1.5),
            TranscriptSegment(text="Second line.", start_time=1.5, end_time=3.0),
        ],
        language="en",
        pipeline_state=PipelineState.TRANSCRIBED,
    )


class TestModelCache:
    def setup_method(self):
        service.reset_caches()

    def teardown_method(self):
        service.reset_caches()

    def test_same_model_loaded_once(self, monkeypatch):
        loads = []

        def fake_load(name, device=None):
            loads.append(name)
            return f"model:{name}"

        monkeypatch.setattr(service.whisper, "load_model", fake_load)
        a = service.get_whisper_model("base", "cpu")
        b = service.get_whisper_model("base", "cpu")
        assert a is b
        assert loads == ["base"]  # loaded exactly once

    def test_model_change_evicts_and_reloads(self, monkeypatch):
        loads = []
        collected = []
        monkeypatch.setattr(service.whisper, "load_model", lambda name, device=None: (loads.append(name) or f"model:{name}"))
        monkeypatch.setattr(service.gc, "collect", lambda: collected.append(True))

        service.get_whisper_model("base", "cpu")
        service.get_whisper_model("large-v3", "cpu")
        assert loads == ["base", "large-v3"]
        assert collected  # eviction ran gc.collect at least once


class TestTranscribeFile:
    def setup_method(self):
        service.reset_caches()

    def teardown_method(self):
        service.reset_caches()

    def _patch(self, monkeypatch):
        # Model load is a no-op object; transcription returns a stub document.
        monkeypatch.setattr(service.whisper, "load_model", lambda name, device=None: object())
        monkeypatch.setattr(service, "_resolve_device", lambda logger: "cpu")

        def fake_transcribe(model, audio_path, language, progress_update_interval_seconds, logger, progress_callback=None):
            if progress_callback:
                progress_callback(1.0)
            doc = _stub_doc()
            doc.source_file = str(audio_path)
            return doc

        monkeypatch.setattr("src.pipeline.steps_transcription.transcribe_audio", fake_transcribe)

    def test_transcribes_audio_and_writes_txt(self, monkeypatch, tmp_path):
        self._patch(monkeypatch)
        # point outputs at tmp by overriding config
        cfg = _tmp_config(tmp_path)
        src_audio = tmp_path / "clip.mp3"
        src_audio.write_bytes(b"x")
        monkeypatch.setattr("src.pipeline.steps_ingestion._probe_duration", lambda p: 1.0)  # skip real decode probe

        result = service.transcribe_file(
            src_audio, model="base", language="en", config_path=cfg, sanitize_output=True
        )
        assert result.status == "success"
        assert Path(result.txt_path).exists()
        assert "Hello world." in Path(result.txt_path).read_text(encoding="utf-8")
        assert result.model == "base"

    def test_does_not_mutate_params_yaml(self, monkeypatch, tmp_path):
        self._patch(monkeypatch)
        cfg = _tmp_config(tmp_path)
        params_file = tmp_path / "params.yaml"
        before = params_file.read_text(encoding="utf-8")
        src_audio = tmp_path / "clip.mp3"
        src_audio.write_bytes(b"x")
        monkeypatch.setattr("src.pipeline.steps_ingestion._probe_duration", lambda p: 1.0)
        service.transcribe_file(src_audio, model="large-v3", language="en", config_path=cfg, sanitize_output=True)
        assert params_file.read_text(encoding="utf-8") == before

    def test_unsupported_extension_raises_media_decode_error(self, monkeypatch, tmp_path):
        self._patch(monkeypatch)
        cfg = _tmp_config(tmp_path)
        bad = tmp_path / "file.pdf"
        bad.write_bytes(b"x")
        import pytest

        from src.utils import MediaDecodeError

        with pytest.raises(MediaDecodeError):
            service.transcribe_file(bad, model="base", language="en", config_path=cfg)

    def test_raises_clear_error_when_ffmpeg_unavailable(self, monkeypatch, tmp_path):
        # Root cause of the GUI hang: whisper's load_audio shells out to a bare
        # `ffmpeg`. When neither a system ffmpeg nor the bundled imageio-ffmpeg
        # binary is available, the shared seam must fail fast with an actionable
        # ProcessingError (which app.py surfaces as a clean gr.Error) instead of
        # letting whisper die mid-pipeline with FileNotFoundError -- which, with
        # the broken traceback-logging landmine, left the GUI pinned at 100%.
        from unittest.mock import Mock

        import pytest

        from src.utils import ProcessingError

        self._patch(monkeypatch)
        cfg = _tmp_config(tmp_path)
        src_audio = tmp_path / "clip.mp3"
        src_audio.write_bytes(b"x")
        # Provisioning finds no bundled binary... (Mock so we can prove the seam
        # actually ran the pre-flight -- the whole point of the fix.)
        fake_on_path = Mock(return_value=None)
        monkeypatch.setattr(service, "ensure_ffmpeg_on_path", fake_on_path)
        # ...and there is no system ffmpeg on PATH either.
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setenv("PATH", str(empty))

        with pytest.raises(ProcessingError, match="ffmpeg"):
            service.transcribe_file(src_audio, model="base", language="en", config_path=cfg)

        fake_on_path.assert_called_once()  # the seam ran the ffmpeg pre-flight

    def test_cleanup_failure_keeps_raw_and_warns(self, monkeypatch, tmp_path):
        self._patch(monkeypatch)
        cfg = _tmp_config(tmp_path)
        src_audio = tmp_path / "clip.mp3"
        src_audio.write_bytes(b"x")
        monkeypatch.setattr("src.pipeline.steps_ingestion._probe_duration", lambda p: 1.0)

        from src.utils import ProcessingError

        def boom(*a, **k):
            raise ProcessingError("Ollama down")

        monkeypatch.setattr("src.pipeline.steps_cleanup.format_document_with_speakers", lambda d: "text")
        monkeypatch.setattr(service, "cleanup_with_ollama", boom)
        result = service.transcribe_file(
            src_audio, model="base", language="en", cleanup=True, config_path=cfg, sanitize_output=True
        )
        assert result.status == "warning"
        assert Path(result.txt_path).exists()        # raw transcript saved
        assert result.clean_path is None

    def test_diarize_backend_failure_keeps_transcript_and_warns(self, monkeypatch, tmp_path):
        # Backend creation (model load / HF auth) failing must degrade to a
        # warning with the raw transcript saved -- not raise out of
        # transcribe_file. Mirror of test_cleanup_failure_keeps_raw_and_warns
        # for the diarization eager-load failure point.
        from unittest.mock import MagicMock

        self._patch(monkeypatch)
        cfg = _tmp_config(tmp_path)
        src_audio = tmp_path / "clip.mp3"
        src_audio.write_bytes(b"x")
        monkeypatch.setattr("src.pipeline.steps_ingestion._probe_duration", lambda p: 1.0)
        # diarization config loads fine, but constructing the backend blows up
        monkeypatch.setattr(service, "load_diarization_config", lambda path: MagicMock(enabled=False))

        def boom(*a, **k):
            raise RuntimeError("HF auth failed / model download blocked")

        monkeypatch.setattr(service, "get_diarization_backend", boom)

        result = service.transcribe_file(
            src_audio, model="base", language="en", diarize=True,
            config_path=cfg, sanitize_output=True,
        )
        assert result.status == "warning"  # degraded, not failed
        assert Path(result.txt_path).exists()  # raw transcript still saved
        # Durable proof diarization didn't run: no diarization metadata, no
        # speakers. (pipeline_state is EXPORTED here -- OutputStep clobbers it.)
        assert result.document.metadata.get("diarization") is None
        assert not result.document.speakers
        blob = " ".join(result.warnings + [result.message or ""]).lower()
        assert "diar" in blob

    def test_successful_diarization_reports_success(self, monkeypatch, tmp_path):
        # Regression guard: a SUCCESSFUL diarization must report status
        # "success", not a false "warning". The degradation signal keys on
        # metadata["diarization"] (written only on success, durable across
        # OutputStep), NOT on pipeline_state (which OutputStep clobbers to
        # EXPORTED regardless -- so a state-based check would warn even on
        # success).
        from unittest.mock import MagicMock

        self._patch(monkeypatch)
        cfg = _tmp_config(tmp_path)
        src_audio = tmp_path / "clip.mp3"
        src_audio.write_bytes(b"x")
        monkeypatch.setattr("src.pipeline.steps_ingestion._probe_duration", lambda p: 1.0)
        monkeypatch.setattr(service, "load_diarization_config", lambda path: MagicMock(enabled=False))
        monkeypatch.setattr(service, "get_diarization_backend", lambda *a, **k: object())

        def fake_diarize(*, document, **kwargs):
            document.pipeline_state = PipelineState.DIARIZED
            document.metadata["diarization"] = {"backend": "mock", "num_speakers_detected": 2}
            return document

        monkeypatch.setattr("src.pipeline.steps_diarization.diarize_document", fake_diarize)

        result = service.transcribe_file(
            src_audio, model="base", language="en", diarize=True,
            config_path=cfg, sanitize_output=True,
        )
        assert result.status == "success"  # no false warning on success
        assert result.document.metadata.get("diarization") is not None

    def test_diarize_step_failure_keeps_transcript_and_warns(self, monkeypatch, tmp_path):
        # Path B (vs the backend-setup path A above): backend setup SUCCEEDS, so
        # DiarizationStep IS added, but diarize_document raises INSIDE the step.
        # The step must degrade (warn + continue), and the service must still
        # classify the run as "warning" with the raw transcript saved -- proving
        # the metadata["diarization"]-absent signal works end-to-end for the
        # in-step failure path, not just backend-setup failure.
        from unittest.mock import MagicMock

        self._patch(monkeypatch)
        cfg = _tmp_config(tmp_path)
        src_audio = tmp_path / "clip.mp3"
        src_audio.write_bytes(b"x")
        monkeypatch.setattr("src.pipeline.steps_ingestion._probe_duration", lambda p: 1.0)
        monkeypatch.setattr(service, "load_diarization_config", lambda path: MagicMock(enabled=False))
        monkeypatch.setattr(service, "get_diarization_backend", lambda *a, **k: object())

        def boom(*a, **k):
            raise RuntimeError("pyannote inference crashed mid-run")

        monkeypatch.setattr("src.pipeline.steps_diarization.diarize_document", boom)

        result = service.transcribe_file(
            src_audio, model="base", language="en", diarize=True,
            config_path=cfg, sanitize_output=True,
        )
        assert result.status == "warning"  # degraded inside the step, not failed
        assert Path(result.txt_path).exists()  # raw transcript still saved
        assert result.document.metadata.get("diarization") is None  # never diarized
        blob = " ".join(result.warnings + [result.message or ""]).lower()
        assert "diar" in blob

    def test_auto_detect_passes_no_language_kwarg(self, monkeypatch, tmp_path):
        # Integration: real transcribe_audio + real pipeline; only the Whisper
        # model and the decode probe are mocked. Asserts language=None reaches
        # model.transcribe as "no language kwarg" and the detected language is
        # captured end-to-end. (Does NOT use self._patch — uses the real
        # transcription path so the Task 3 auto-detect fix is exercised here.)
        from unittest.mock import MagicMock

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "hola", "segments": [], "language": "es"}
        monkeypatch.setattr(service.whisper, "load_model", lambda name, device=None: mock_model)
        monkeypatch.setattr(service, "_resolve_device", lambda logger: "cpu")
        # AudioIngestionStep probes decodability (Task 4); force it to pass.
        monkeypatch.setattr("src.pipeline.steps_ingestion._probe_duration", lambda p: 1.0)

        cfg = _tmp_config(tmp_path)
        src_audio = tmp_path / "clip.mp3"
        src_audio.write_bytes(b"x")
        result = service.transcribe_file(
            src_audio, model="base", language=None, config_path=cfg, sanitize_output=True
        )
        _, kwargs = mock_model.transcribe.call_args
        assert "language" not in kwargs  # auto-detect: no language kwarg
        assert result.language == "es"  # detected language captured end-to-end


def _tmp_config(tmp_path) -> str:
    """Write a minimal general_config + params + prompts under tmp_path; return config path."""
    import textwrap

    (tmp_path / "params.yaml").write_text("transcription_model: base\ncleanup_model: x\n", encoding="utf-8")
    (tmp_path / "prompts.yaml").write_text("cleanup_prompt: clean this\n", encoding="utf-8")
    (tmp_path / "diarization.yaml").write_text("enabled: false\n", encoding="utf-8")
    cfg = tmp_path / "general_config.yaml"
    cfg.write_text(
        textwrap.dedent(
            f"""
            paths:
              videos: "{tmp_path}/videos"
              audios: "{tmp_path}/audios"
              transcripts: "{tmp_path}/transcripts"
              logs: "{tmp_path}/logs"
            files:
              params: "{tmp_path}/params.yaml"
              prompts: "{tmp_path}/prompts.yaml"
              diarization: "{tmp_path}/diarization.yaml"
            extensions:
              video: [".mp4", ".mov", ".avi", ".mkv", ".webm"]
              audio: [".mp3", ".m4a"]
            output:
              transcript_extension: ".txt"
              cleaned_suffix: "_clean"
              extracted_audio_extension: ".mp3"
            ollama:
              url: "http://localhost:11434/api/generate"
              timeout_seconds: 600
              request_content_type: "application/json"
            processing:
              progress_update_interval_seconds: 0.25
            dependencies:
              ffmpeg_executable: "ffmpeg"
            logging:
              level: "INFO"
              file_name: "transcriber.log"
              format: "%(message)s"
            """
        ),
        encoding="utf-8",
    )
    return str(cfg)
