from src.transcription.diarization_config import load_diarization_config


def test_load_diarization_config():
    config = load_diarization_config("configurations/diarization.yaml")
    assert config.backend == "pyannote"
    assert config.min_speakers == 2
    assert config.max_speakers == 10
    assert config.key_speakers == 3
    assert config.enabled is False
