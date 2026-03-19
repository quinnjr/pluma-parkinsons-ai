# tests/test_models.py
import pytest
from src.models import Subject, OmicsProfile, BiomarkerHit, Stage1Output

def test_subject_creation():
    s = Subject(subject_id="PD_001", diagnosis="PD", age=65, sex="M")
    assert s.subject_id == "PD_001"
    assert s.diagnosis == "PD"

def test_subject_diagnosis_validation():
    with pytest.raises(ValueError):
        Subject(subject_id="X", diagnosis="INVALID", age=50, sex="F")

def test_omics_profile_stores_features():
    profile = OmicsProfile(modality="genomics", features={"LRRK2_p.G2019S": 1.0})
    assert profile.features["LRRK2_p.G2019S"] == 1.0

def test_stage1_output_serializes_to_json():
    output = Stage1Output(
        subject_id="PD_001",
        diagnosis="PD",
        prediction_confidence=0.91,
        disease_stage="early",
        top_biomarkers=[
            BiomarkerHit(modality="genomics", feature="LRRK2_p.G2019S", shap_value=0.34)
        ],
        mofa_factors={},
        environmental_risk_score=6.8,
    )
    data = output.to_dict()
    assert data["subject_id"] == "PD_001"
    assert data["top_biomarkers"][0]["feature"] == "LRRK2_p.G2019S"

def test_stage1_output_roundtrip():
    output = Stage1Output(
        subject_id="HC_001",
        diagnosis="HC",
        prediction_confidence=0.05,
        disease_stage=None,
        top_biomarkers=[],
        mofa_factors={"factor_1": 0.12},
        environmental_risk_score=2.1,
    )
    data = output.to_dict()
    restored = Stage1Output.from_dict(data)
    assert restored.subject_id == output.subject_id
    assert restored.prediction_confidence == output.prediction_confidence
