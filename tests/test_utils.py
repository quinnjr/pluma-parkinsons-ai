# tests/test_utils.py
import yaml

from src.utils import ensure_dir, load_config, load_jsonl, save_jsonl


def test_load_config(tmp_path):
    cfg = {"key": "value", "nested": {"a": 1}}
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml.dump(cfg))
    loaded = load_config(str(config_file))
    assert loaded["key"] == "value"
    assert loaded["nested"]["a"] == 1

def test_ensure_dir_creates_path(tmp_path):
    new_dir = tmp_path / "a" / "b" / "c"
    result = ensure_dir(new_dir)
    assert result.exists()
    assert result.is_dir()

def test_save_and_load_jsonl(tmp_path):
    records = [{"id": 1, "val": "a"}, {"id": 2, "val": "b"}]
    path = tmp_path / "test.jsonl"
    save_jsonl(records, path)
    loaded = load_jsonl(path)
    assert loaded == records
