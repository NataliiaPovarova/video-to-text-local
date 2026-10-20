import gc
import logging
import weakref
from pathlib import Path
from types import SimpleNamespace

import src.service as service
from src.models import PipelineState, TranscriptDocument, TranscriptSegment
from src.utils.ffmpeg_tools import MediaInfo

AUDIO_INFO = MediaInfo(True, True, 1.0, 1, 0, "mp3", 44100, "mono")
VIDEO_INFO = MediaInfo(True, True, 1.0, 1, 1, "aac", 44100, "mono")
MUTE_VIDEO_INFO = MediaInfo(True, False, 1.0, 0, 1, None, None, None)


def _stub_probe(monkeypatch, info=AUDIO_INFO):
    """Replace the seam's ffmpeg probe (tests use fake bytes, not real media)."""
    monkeypatch.setattr(service, "probe_media", lambda path, ffmpeg="ffmpeg": info)


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


class _FakeModel:
    """Stand-in for a GPU-resident model; weakref-able so a test can prove it
    was actually freed (no lingering reference), not just dropped from a cache."""

    def __init__(self, name):
        self.name = name


class TestGpuModelSwap:
    # Issue #27: on an 8 GB GPU, Whisper large-v3 and pyannote co-resident in
    # VRAM overflow into shared system memory (the Windows driver spills
    # instead of raising OOM) and diarization slows ~15x. On CUDA at most one
    # heavy model may be resident at a time; on CPU both stay cached.
    _CFG = SimpleNamespace(backend="pyannote", model="pyannote/speaker-diarization-3.1")

    def setup_method(self):
        service.reset_caches()

    def teardown_method(self):
        service.reset_caches()

    def test_cuda_pyannote_load_frees_whisper(self, monkeypatch):
        refs = []

        def fake_load(name, device=None):
            model = _FakeModel(name)
            refs.append(weakref.ref(model))
            return model

        alive_at_load = []

        def fake_create(config, device, logger):
            gc.collect()
            alive_at_load.append(refs[0]() is not None)
            return _FakeModel("pyannote")

        monkeypatch.setattr(service.whisper, "load_model", fake_load)
        monkeypatch.setattr(service, "create_diarization_backend", fake_create)
        monkeypatch.setattr(service, "_empty_cuda_cache", lambda: None)
        log = logging.getLogger("test")

        service.get_whisper_model("large-v3", "cuda", log)  # result not kept
        service.get_diarization_backend(self._CFG, "cuda", log)
        assert alive_at_load == [False]

    def test_cuda_whisper_load_frees_pyannote(self, monkeypatch):
        refs = []

        def fake_create(config, device, logger):
            backend = _FakeModel("pyannote")
            refs.append(weakref.ref(backend))
            return backend

        alive_at_load = []

        def fake_load(name, device=None):
            gc.collect()
            alive_at_load.append(refs[0]() is not None)
            return _FakeModel(name)

        monkeypatch.setattr(service.whisper, "load_model", fake_load)
        monkeypatch.setattr(service, "create_diarization_backend", fake_create)
        monkeypatch.setattr(service, "_empty_cuda_cache", lambda: None)
        log = logging.getLogger("test")

        service.get_diarization_backend(self._CFG, "cuda", log)  # result not kept
        service.get_whisper_model("large-v3", "cuda", log)
        assert alive_at_load == [False]

    def test_cpu_keeps_both_models_cached(self, monkeypatch):
        # Regression guard: RAM is plentiful on CPU, so no swapping -- a batch
        # must not reload either model per file.
        loads, creates = [], []
        monkeypatch.setattr(service.whisper, "load_model", lambda name, device=None: (loads.append(name) or _FakeModel(name)))
        monkeypatch.setattr(service, "create_diarization_backend", lambda c, d, l: (creates.append(1) or _FakeModel("pyannote")))
        log = logging.getLogger("test")

        for _ in range(2):
            service.get_whisper_model("base", "cpu", log)
            service.get_diarization_backend(self._CFG, "cpu", log)
        assert loads == ["base"]
        assert creates == [1]

    def test_cuda_job_loads_pyannote_after_transcription_with_whisper_freed(self, monkeypatch, tmp_path):
        # End-to-end through the seam: pyannote must not be loaded up-front
        # (it was, at service setup, before the pipeline ran), and by the time
        # it loads no reference to the Whisper model may survive -- neither in
        # transcribe_file's locals nor in TranscriptionStep.
        from unittest.mock import MagicMock

        events, refs, alive_at_load = [], [], []

        def fake_load(name, device=None):
            events.append("load_whisper")
            model = _FakeModel(name)
            refs.append(weakref.ref(model))
            return model

        def fake_transcribe(model, audio_path, language, progress_update_interval_seconds, logger, progress_callback=None):
            events.append("transcribe")
            doc = _stub_doc()
            doc.source_file = str(audio_path)
            return doc

        def fake_create(config, device, logger):
            gc.collect()
            events.append("load_pyannote")
            alive_at_load.append(any(r() is not None for r in refs))
            return _FakeModel("pyannote")

        def fake_diarize(*, document, **kwargs):
            document.metadata["diarization"] = {"backend": "mock", "num_speakers_detected": 1}
            return document

        monkeypatch.setattr(service.whisper, "load_model", fake_load)
        monkeypatch.setattr(service, "_resolve_device", lambda logger: "cuda")
        monkeypatch.setattr(service, "_empty_cuda_cache", lambda: None)
        monkeypatch.setattr("src.pipeline.steps_transcription.transcribe_audio", fake_transcribe)
        monkeypatch.setattr("src.pipeline.steps_ingestion._probe_duration", lambda p: 1.0)
        monkeypatch.setattr(
            service, "load_diarization_config",
            lambda path: MagicMock(enabled=False, backend="pyannote", model="m"),
        )
        monkeypatch.setattr(service, "create_diarization_backend", fake_create)
        monkeypatch.setattr("src.pipeline.steps_diarization.diarize_document", fake_diarize)

        cfg = _tmp_config(tmp_path)
        src_audio = tmp_path / "clip.mp3"
        src_audio.write_bytes(b"x")
        result = service.transcribe_file(
            src_audio, model="large-v3", language="en", diarize=True,
            config_path=cfg, sanitize_output=True,
        )
        assert result.status == "success"
        assert events == ["load_whisper", "transcribe", "load_pyannote"]
        assert alive_at_load == [False]

    def _patch_real_asr(self, monkeypatch, device="cuda"):
        # Real transcribe_audio (with its progress ticker thread running) over a
        # weakref-able fake Whisper; records, at every model load, whether the
        # OTHER model is still alive.
        from unittest.mock import MagicMock

        class FakeWhisper(_FakeModel):
            def transcribe(self, audio, **kw):
                return {"text": "hi", "language": "en", "segments": [{"text": "hi", "start": 0.0, "end": 1.0}]}

        wrefs, brefs, events = [], [], []

        def fake_load(name, device=None):
            gc.collect()
            events.append(("load_whisper", any(r() is not None for r in brefs)))
            model = FakeWhisper(name)
            wrefs.append(weakref.ref(model))
            return model

        def fake_create(config, device, logger):
            gc.collect()
            events.append(("load_pyannote", any(r() is not None for r in wrefs)))
            backend = _FakeModel("pyannote")
            brefs.append(weakref.ref(backend))
            return backend

        def fake_diarize(*, document, **kwargs):
            document.metadata["diarization"] = {"backend": "mock", "num_speakers_detected": 1}
            return document

        monkeypatch.setattr(service.whisper, "load_model", fake_load)
        monkeypatch.setattr(service, "_resolve_device", lambda logger: device)
        monkeypatch.setattr(service, "_empty_cuda_cache", lambda: None)
        monkeypatch.setattr("src.pipeline.steps_ingestion._probe_duration", lambda p: 1.0)
        monkeypatch.setattr("src.transcription.asr_engine._get_audio_duration_seconds", lambda p: 2.0)
        monkeypatch.setattr(
            service, "load_diarization_config",
            lambda path: MagicMock(enabled=False, backend="pyannote", model="m"),
        )
        monkeypatch.setattr(service, "create_diarization_backend", fake_create)
        monkeypatch.setattr("src.pipeline.steps_diarization.diarize_document", fake_diarize)
        return events, wrefs

    def test_cuda_batch_swaps_models_without_pinning_either(self, monkeypatch, tmp_path):
        events, _ = self._patch_real_asr(monkeypatch)
        cfg = _tmp_config(tmp_path)
        for n in ("a", "b", "c"):
            p = tmp_path / f"{n}.mp3"
            p.write_bytes(b"x")
            result = service.transcribe_file(p, model="large-v3", diarize=True, config_path=cfg, progress_callback=lambda f: None)
            assert result.status == "success"
        # (event, other-model-still-alive): neither model is ever pinned.
        assert events == [("load_whisper", False), ("load_pyannote", False)] * 3

    def test_cuda_empty_cache_runs_after_whisper_refs_dropped(self, monkeypatch, tmp_path):
        # torch.cuda.empty_cache() only returns blocks whose tensors are dead;
        # it must run after the last Whisper reference is gone.
        _, wrefs = self._patch_real_asr(monkeypatch)
        seen = []
        monkeypatch.setattr(service, "_empty_cuda_cache", lambda: seen.append([r() is None for r in wrefs]))
        cfg = _tmp_config(tmp_path)
        p = tmp_path / "a.mp3"
        p.write_bytes(b"x")
        service.transcribe_file(p, model="large-v3", diarize=True, config_path=cfg)
        assert seen and seen[-1] == [True]  # the call before pyannote loads


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
        _stub_probe(monkeypatch)  # skip real ffmpeg probe

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
        _stub_probe(monkeypatch)
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

    def test_transcribes_aac_audio(self, monkeypatch, tmp_path):
        self._patch(monkeypatch)
        cfg = _tmp_config(tmp_path)
        src_audio = tmp_path / "clip.aac"
        src_audio.write_bytes(b"x")
        _stub_probe(monkeypatch)

        result = service.transcribe_file(
            src_audio, model="base", language="en", config_path=cfg, sanitize_output=True
        )
        assert result.status == "success"
        assert Path(result.txt_path).exists()

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
        _stub_probe(monkeypatch)

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
        # for the backend-load failure point (lazy, inside DiarizationStep
        # since issue #27).
        from unittest.mock import MagicMock

        self._patch(monkeypatch)
        cfg = _tmp_config(tmp_path)
        src_audio = tmp_path / "clip.mp3"
        src_audio.write_bytes(b"x")
        _stub_probe(monkeypatch)
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
        _stub_probe(monkeypatch)
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
        # Path B (vs the backend-load path A above): backend load SUCCEEDS, but
        # diarize_document raises INSIDE the step.
        # The step must degrade (warn + continue), and the service must still
        # classify the run as "warning" with the raw transcript saved -- proving
        # the metadata["diarization"]-absent signal works end-to-end for the
        # in-step failure path, not just backend-setup failure.
        from unittest.mock import MagicMock

        self._patch(monkeypatch)
        cfg = _tmp_config(tmp_path)
        src_audio = tmp_path / "clip.mp3"
        src_audio.write_bytes(b"x")
        _stub_probe(monkeypatch)
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
        _stub_probe(monkeypatch)

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
    # Forward slashes: a Windows path (C:\Users\...) inside a double-quoted YAML
    # scalar is parsed as escape sequences (\U...) and fails to load.
    root = tmp_path.as_posix()
    cfg.write_text(
        textwrap.dedent(
            f"""
            paths:
              videos: "{root}/videos"
              audios: "{root}/audios"
              transcripts: "{root}/transcripts"
              logs: "{root}/logs"
            files:
              params: "{root}/params.yaml"
              prompts: "{root}/prompts.yaml"
              diarization: "{root}/diarization.yaml"
            extensions:
              video: [".mp4", ".mov", ".avi", ".mkv", ".webm"]
              audio: [".mp3", ".m4a", ".aac"]
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


class TestInputFormatGate:
    """C-006 / plan §4.4: the seam accepts a file because ffmpeg finds audio in
    it, not because of its extension; video vs audio is decided by the probe."""

    def setup_method(self):
        service.reset_caches()

    def teardown_method(self):
        service.reset_caches()

    def _run(self, monkeypatch, tmp_path, name, info):
        TestTranscribeFile()._patch(monkeypatch)
        _stub_probe(monkeypatch, info)
        src = tmp_path / name
        src.write_bytes(b"x")
        return service.transcribe_file(src, model="base", language="en", config_path=_tmp_config(tmp_path))

    def test_wav_is_accepted(self, monkeypatch, tmp_path):
        # The reported bug: CLI discovery found clip.wav, the seam rejected it.
        assert self._run(monkeypatch, tmp_path, "clip.wav", AUDIO_INFO).status == "success"

    def test_extension_does_not_decide_acceptance(self, monkeypatch, tmp_path):
        assert self._run(monkeypatch, tmp_path, "clip.opus", AUDIO_INFO).status == "success"

    def test_no_audio_track_raises_media_decode_error(self, monkeypatch, tmp_path):
        import pytest

        from src.utils import MediaDecodeError

        with pytest.raises(MediaDecodeError, match="no audio track"):
            self._run(monkeypatch, tmp_path, "silent.mp4", MUTE_VIDEO_INFO)

    def test_unreadable_raises_media_decode_error(self, monkeypatch, tmp_path):
        import pytest

        from src.utils import MediaDecodeError

        with pytest.raises(MediaDecodeError, match="Could not read"):
            self._run(monkeypatch, tmp_path, "clip.mp3", MediaInfo.unreadable())

    def test_audio_only_container_is_not_extracted(self, monkeypatch, tmp_path):
        # .mkv holding only audio: the probe says audio -> no video extraction.
        def no_video_step(*a, **k):
            raise AssertionError("VideoIngestionStep must not be used for audio-only media")

        monkeypatch.setattr(service, "VideoIngestionStep", no_video_step)
        assert self._run(monkeypatch, tmp_path, "music.mkv", AUDIO_INFO).status == "success"

    def test_video_is_extracted_even_with_audio_extension(self, monkeypatch, tmp_path):
        calls = []

        def fake_extract(src, out, ffmpeg="ffmpeg"):
            calls.append(Path(src).name)
            Path(out).write_bytes(b"audio")
            return Path(out)

        monkeypatch.setattr("src.pipeline.steps_ingestion.extract_audio", fake_extract)
        assert self._run(monkeypatch, tmp_path, "weird.mp3", VIDEO_INFO).status == "success"
        assert calls == ["weird.mp3"]

    def test_steps_reuse_the_seam_probe(self, monkeypatch, tmp_path):
        # One probe per file: ingestion steps read context.media, not ffmpeg again.
        def must_not_probe(*a, **k):
            raise AssertionError("ingestion re-probed the file")

        monkeypatch.setattr("src.pipeline.steps_ingestion.probe_media", must_not_probe)
        assert self._run(monkeypatch, tmp_path, "clip.wav", AUDIO_INFO).status == "success"


class TestInputFormatGateRealFfmpeg:
    """Same gate against real files generated by ffmpeg (no stubbed probe)."""

    def _make(self, path, *args):
        import shutil
        import subprocess

        import pytest

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            pytest.skip("no system ffmpeg")
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *args, str(path)], check=True)
        return path

    def test_real_wav_flac_ogg_are_transcribed(self, monkeypatch, tmp_path):
        TestTranscribeFile()._patch(monkeypatch)
        cfg = _tmp_config(tmp_path)
        for name in ("clip.wav", "clip.flac", "clip.ogg"):
            src = self._make(tmp_path / name, "-f", "lavfi", "-i", "sine=duration=1")
            assert service.transcribe_file(src, model="base", language="en", config_path=cfg).status == "success", name

    def test_real_mute_video_is_rejected(self, monkeypatch, tmp_path):
        import pytest

        from src.utils import MediaDecodeError

        TestTranscribeFile()._patch(monkeypatch)
        src = self._make(tmp_path / "mute.mp4", "-f", "lavfi", "-i", "testsrc=duration=1:size=64x48:rate=5")
        with pytest.raises(MediaDecodeError, match="no audio track"):
            service.transcribe_file(src, model="base", language="en", config_path=_tmp_config(tmp_path))


def test_config_lists_common_formats():
    # One extension list (config) feeds CLI discovery, the GUI picker and the
    # transcribe wrapper; it is a convenience list, the probe is the gate.
    from src.utils import load_yaml_file

    ext = load_yaml_file("configurations/general_config.yaml")["extensions"]
    for e in (".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus"):
        assert e in ext["audio"], e
    for e in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
        assert e in ext["video"], e


def test_seam_has_no_hardcoded_extension_lists():
    assert not hasattr(service, "_VIDEO_EXTS") and not hasattr(service, "_AUDIO_EXTS")
