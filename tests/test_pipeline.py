# tests/test_pipeline.py
import pytest
import yaml
from src.pipeline import Pipeline, STAGES

def test_pipeline_stages_list():
    assert "acquire" in STAGES
    assert "preprocess" in STAGES
    assert "integrate" in STAGES
    assert "build_instructions" in STAGES
    assert "train" in STAGES

def test_pipeline_initializes_from_config(tmp_path):
    acq_cfg = {
        "ppmi": {"data_dir": str(tmp_path / "ppmi"), "modalities": []},
        "geo": {"data_dir": str(tmp_path / "geo"), "pd_search_terms": [], "max_results": 5},
        "environmental": {
            "epa_aqs": {"base_url": "http://x", "data_dir": str(tmp_path)},
            "usgs_pesticides": {"data_dir": str(tmp_path)},
            "nhanes": {"data_dir": str(tmp_path), "cycles": []},
        },
    }
    cfg_path = tmp_path / "acquisition.yaml"
    cfg_path.write_text(yaml.dump(acq_cfg))
    pipeline = Pipeline(acquisition_config=str(cfg_path))
    assert pipeline is not None

def test_pipeline_rejects_unknown_stage(tmp_path):
    acq_cfg = {
        "ppmi": {"data_dir": str(tmp_path), "modalities": []},
        "geo": {"data_dir": str(tmp_path), "pd_search_terms": [], "max_results": 5},
        "environmental": {
            "epa_aqs": {"base_url": "http://x", "data_dir": str(tmp_path)},
            "usgs_pesticides": {"data_dir": str(tmp_path)},
            "nhanes": {"data_dir": str(tmp_path), "cycles": []},
        },
    }
    cfg_path = tmp_path / "acquisition.yaml"
    cfg_path.write_text(yaml.dump(acq_cfg))
    pipeline = Pipeline(acquisition_config=str(cfg_path))
    with pytest.raises(ValueError, match="Unknown stage"):
        pipeline.run_stage("nonexistent_stage")
