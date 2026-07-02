from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.transcription.diarization_config import DiarizationConfig
from src.transcription.diarization_backends.pyannote_backend import (
    PyannoteBackend,
    create_pyannote_backend,
)


class TestPyannoteBackend:
    # `diarize` decodes audio into an in-memory waveform mapping and hands THAT
    # to the pipeline (the torchcodec-bypass path), then reads the pyannote >= 4.0
    # `output.speaker_diarization` wrapper. These tests patch `_load_in_memory`
    # so no ffmpeg/audio file is touched, and drive the pipeline via that wrapper.
    @patch.object(PyannoteBackend, "_load_in_memory")
    @patch("src.transcription.diarization_backends.pyannote_backend.ProgressHook")
    def test_diarize_parses_itertracks(self, mock_progress_hook, mock_load):
        file_input = {"waveform": "WAVE", "sample_rate": 16000, "uri": "audio"}
        mock_load.return_value = file_input

        pipeline = MagicMock()
        segment_a = MagicMock(start=0.0, end=5.0)
        segment_b = MagicMock(start=5.0, end=10.0)
        pipeline.return_value.speaker_diarization.itertracks.return_value = [
            (segment_a, None, "SPEAKER_00"),
            (segment_b, None, "SPEAKER_01"),
        ]
        mock_hook = MagicMock()
        mock_progress_hook.return_value.__enter__.return_value = mock_hook

        backend = PyannoteBackend(pipeline=pipeline, model_id="test-model")
        turns = backend.diarize(Path("audio.wav"), min_speakers=2, max_speakers=5)

        assert len(turns) == 2
        assert turns[0].speaker_id == "spk_0"
        assert turns[1].speaker_id == "spk_1"
        assert (turns[0].start_time, turns[0].end_time) == (0.0, 5.0)
        assert (turns[1].start_time, turns[1].end_time) == (5.0, 10.0)
        # The decoded waveform mapping is what reaches the pipeline, not the path.
        mock_load.assert_called_once_with(Path("audio.wav"), logger=None)
        pipeline.assert_called_once_with(
            file_input, hook=mock_hook, min_speakers=2, max_speakers=5
        )
        pipeline.return_value.speaker_diarization.itertracks.assert_called_once_with(
            yield_label=True
        )

    @patch.object(PyannoteBackend, "_load_in_memory")
    @patch("src.transcription.diarization_backends.pyannote_backend.ProgressHook")
    def test_diarize_with_num_speakers(self, mock_progress_hook, mock_load):
        file_input = {"waveform": "WAVE", "sample_rate": 16000, "uri": "audio"}
        mock_load.return_value = file_input

        pipeline = MagicMock()
        pipeline.return_value.speaker_diarization.itertracks.return_value = []
        mock_hook = MagicMock()
        mock_progress_hook.return_value.__enter__.return_value = mock_hook

        backend = PyannoteBackend(pipeline=pipeline, model_id="test-model")
        turns = backend.diarize(Path("audio.wav"), num_speakers=2)

        assert turns == []
        mock_load.assert_called_once_with(Path("audio.wav"), logger=None)
        # num_speakers is exclusive: min/max must NOT be forwarded alongside it.
        pipeline.assert_called_once_with(file_input, hook=mock_hook, num_speakers=2)

    @patch("src.transcription.diarization_backends.pyannote_backend.Pipeline")
    @patch("src.transcription.diarization_backends.pyannote_backend.torch.device")
    def test_create_pyannote_backend(self, mock_device, mock_pipeline_cls):
        mock_pipeline = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline

        config = DiarizationConfig(
            enabled=True,
            backend="pyannote",
            model="pyannote/speaker-diarization-3.1",
            hf_token_env="HF_TOKEN",
            num_speakers=None,
            min_speakers=2,
            max_speakers=10,
            key_speakers=3,
            overlap_threshold=0.2,
            min_segment_duration=0.3,
            progress_update_interval_seconds=0.25,
        )

        with patch.dict("os.environ", {"HF_TOKEN": "test-token"}):
            backend = create_pyannote_backend(config, "cpu", MagicMock())

        assert backend.name == "pyannote"
        mock_pipeline.to.assert_called_once()
