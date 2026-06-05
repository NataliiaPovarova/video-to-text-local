from src.transcription.diarization_backends.base import normalize_speaker_id


class TestNormalizeSpeakerId:
    def test_pyannote_label(self):
        assert normalize_speaker_id("SPEAKER_00") == "spk_0"
        assert normalize_speaker_id("SPEAKER_12") == "spk_12"

    def test_already_normalized(self):
        assert normalize_speaker_id("spk_1") == "spk_1"
