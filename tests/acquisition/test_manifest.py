import pytest
import pandas as pd
from src.acquisition.manifest import SampleManifest

def test_manifest_add_and_retrieve():
    m = SampleManifest()
    m.add_subject("PD_001", diagnosis="PD", age=65, sex="M", cohort="PPMI")
    assert "PD_001" in m
    assert m.get("PD_001")["diagnosis"] == "PD"

def test_manifest_link_external_id():
    m = SampleManifest()
    m.add_subject("PD_001", diagnosis="PD", age=65, sex="M", cohort="PPMI")
    m.link_external_id("PD_001", source="GEO", external_id="GSM123456")
    assert m.resolve_external("GEO", "GSM123456") == "PD_001"

def test_manifest_save_load(tmp_path):
    m = SampleManifest()
    m.add_subject("PD_001", diagnosis="PD", age=65, sex="M", cohort="PPMI")
    m.add_subject("HC_001", diagnosis="HC", age=60, sex="F", cohort="PPMI")
    path = tmp_path / "manifest.csv"
    m.save(path)
    m2 = SampleManifest.load(path)
    assert "PD_001" in m2
    assert m2.get("HC_001")["diagnosis"] == "HC"

def test_manifest_filter_by_diagnosis():
    m = SampleManifest()
    m.add_subject("PD_001", diagnosis="PD", age=65, sex="M", cohort="PPMI")
    m.add_subject("HC_001", diagnosis="HC", age=60, sex="F", cohort="PPMI")
    pd_subjects = m.filter(diagnosis="PD")
    assert len(pd_subjects) == 1
    assert pd_subjects[0] == "PD_001"

def test_manifest_rejects_invalid_diagnosis():
    m = SampleManifest()
    with pytest.raises(ValueError):
        m.add_subject("X_001", diagnosis="UNKNOWN", age=50, sex="M", cohort="PPMI")

def test_resolve_nonexistent_external_id_returns_none():
    m = SampleManifest()
    assert m.resolve_external("GEO", "NONEXISTENT") is None
