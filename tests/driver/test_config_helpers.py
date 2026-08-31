"""Fast unit tests for driver config helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hispecdrp.driver.config import _get_nested, get_config_filepath, load_config_file


def test_get_nested_dict_and_attr():
    meta = {"a": {"b": 3}}
    assert _get_nested(meta, "a.b") == 3
    assert _get_nested(meta, "a.missing") is None
    assert _get_nested(None, "a.b") is None

    class Nest:
        def __init__(self):
            self.inner = type("Inner", (), {"value": 7})()

    assert _get_nested(Nest(), "inner.value") == 7
    assert _get_nested(Nest(), "inner.missing") is None


def test_get_config_filepath_and_load_yaml(tmp_path):
    cfg = tmp_path / "toy_config.yaml"
    payload = {"class": "Stage0Pipeline", "primitives": {}}
    cfg.write_text(yaml.safe_dump(payload), encoding="utf-8")

    resolved = get_config_filepath(str(cfg))
    assert Path(resolved).is_file()
    assert load_config_file(resolved)["class"] == "Stage0Pipeline"

    with pytest.raises(FileNotFoundError):
        get_config_filepath(str(tmp_path / "does_not_exist.yaml"))

    bad = tmp_path / "toy_config.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        load_config_file(str(bad))


def test_stage2_pipeline_config_packaged():
    path = get_config_filepath("Stage2Pipeline.yaml")
    cfg = load_config_file(path)
    assert cfg["class"] == "Stage2Pipeline"
