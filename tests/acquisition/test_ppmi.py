# tests/acquisition/test_ppmi.py
import pytest
import pandas as pd
from pathlib import Path
from src.acquisition.ppmi import PPMIClient

def test_client_initializes_with_data_dir(tmp_path):
    client = PPMIClient(data_dir=str(tmp_path))
    assert client.data_dir == tmp_path

def test_diagnosis_mapping():
    client = PPMIClient(data_dir="data/raw/ppmi")
    assert client._map_diagnosis("Parkinson's Disease") == "PD"
    assert client._map_diagnosis("Healthy Control") == "HC"
    assert client._map_diagnosis("SWEDD") == "SWEDD"
    assert client._map_diagnosis("Prodromal") == "Prodromal"

def test_diagnosis_mapping_raises_for_unknown():
    client = PPMIClient(data_dir="data/raw/ppmi")
    with pytest.raises(ValueError):
        client._map_diagnosis("Unknown Cohort")

def test_build_manifest_from_csv(tmp_path):
    csv_path = tmp_path / "Demographics.csv"
    pd.DataFrame({
        "PATNO": ["001", "002"],
        "COHORT_DEFINITION": ["Parkinson's Disease", "Healthy Control"],
        "AGE_AT_VISIT": [65, 60],
        "SEX": ["Male", "Female"],
    }).to_csv(csv_path, index=False)

    client = PPMIClient(data_dir=str(tmp_path))
    manifest = client.build_manifest_from_csv(csv_path)
    assert "PPMI_001" in manifest
    assert manifest.get("PPMI_001")["diagnosis"] == "PD"
    assert manifest.get("PPMI_002")["diagnosis"] == "HC"
    assert manifest.get("PPMI_002")["sex"] == "F"

def test_list_available_files(tmp_path):
    (tmp_path / "Demographics.csv").write_text("col\nval")
    (tmp_path / "MRI_data.csv").write_text("col\nval")
    client = PPMIClient(data_dir=str(tmp_path))
    files = client.list_available_files()
    assert len(files) == 2
    assert all(f.suffix == ".csv" for f in files)
