import yaml

from .errors import ProcessingError


def load_yaml_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ProcessingError(f"Expected YAML mapping in '{path}', got {type(data).__name__}.")
    return data
