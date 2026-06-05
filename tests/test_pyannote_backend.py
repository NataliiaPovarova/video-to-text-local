from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.transcription.diarization_config import DiarizationConfig
from src.transcription.diarization_backends.pyannote_backend import (
    PyannoteBackend,
    create_pyannote_backend,
)


class TestPyannoteBackend:
    @patch("src.transcription.diarization_backends.pyannote_backend.ProgressHook")
    def test_diarize_parses_itertracks(self, mock_progress_hook):
        pipeline = MagicMock()
        segment_a = MagicMock(start=0.0, end=5.0)
        segment_b = MagicMock(start=5.0, end=10.0)
        pipeline.return_value.itertracks.return_value = [
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
        pipeline.assert_called_once_with(
            "audio.wav", hook=mock_hook, min_speakers=2, max_speakers=5
        )

    @patch("src.transcription.diarization_backends.pyannote_backend.ProgressHook")
    def test_diarize_with_num_speakers(self, mock_progress_hook):
        pipeline = MagicMock()
        pipeline.return_value.itertracks.return_value = []
        mock_hook = MagicMock()
        mock_progress_hook.return_value.__enter__.return_value = mock_hook

        backend = PyannoteBackend(pipeline=pipeline, model_id="test-model")
        backend.diarize(Path("audio.wav"), num_speakers=2)

        pipeline.assert_called_once_with("audio.wav", hook=mock_hook, num_speakers=2)

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
