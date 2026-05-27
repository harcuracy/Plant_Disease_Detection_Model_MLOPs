from pathlib import Path
from typing import Any

import yaml

from cnnClassifier import logger


class ConfigBox(dict):
    def __init__(self, mapping: dict[str, Any]):
        super().__init__()
        for key, value in mapping.items():
            self[key] = self._convert(value)

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    @classmethod
    def _convert(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls._convert(item) for item in value]
        return value


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    with open(path_to_yaml, "r", encoding="utf-8") as yaml_file:
        content = yaml.safe_load(yaml_file) or {}
    logger.info(f"YAML file loaded: {path_to_yaml}")
    return ConfigBox(content)


def create_directories(path_to_directories: list[Path]) -> None:
    for path in path_to_directories:
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")


def get_size(path: Path) -> str:
    size_in_kb = round(path.stat().st_size / 1024)
    return f"~ {size_in_kb} KB"


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as yaml_file:
        yaml.safe_dump(data, yaml_file, sort_keys=False)
    logger.info(f"YAML file saved: {path}")
