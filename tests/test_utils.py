from pathlib import Path

import pytest

from src.utils.config import load_yaml_file
from src.utils.errors import ProcessingError


class TestLoadYamlFile:
    def test_loads_valid_yaml(self, tmp_path: Path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("key: value\nnested:\n  a: 1\n", encoding="utf-8")
        result = load_yaml_file(str(yaml_file))
        assert result == {"key": "value", "nested": {"a": 1}}

    def test_raises_on_non_mapping(self, tmp_path: Path):
        yaml_file = tmp_path / "list.yaml"
        yaml_file.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ProcessingError, match="Expected YAML mapping"):
            load_yaml_file(str(yaml_file))

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_yaml_file("/nonexistent/path.yaml")


class TestProcessingError:
    def test_is_runtime_error(self):
        exc = ProcessingError("test")
        assert isinstance(exc, RuntimeError)
        assert str(exc) == "test"
