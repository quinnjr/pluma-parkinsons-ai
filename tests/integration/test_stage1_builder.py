# tests/integration/test_stage1_builder.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from src.models import Stage1Output, BiomarkerHit
from src.integration.stage1_builder import Stage1Builder

@pytest.fixture
def mock_ensemble():
    ens = MagicMock()
    ens.predict_proba.return_value = np.array([0.91, 0.08])
    ens.compute_shap.return_value = np.array([[0.34, -0.1, 0.21],
                                              [-0.05, 0.02, -0.01]])
    ens._feature_names = ["LRRK2_p.G2019S", "GBA_N370S", "SNCA_expr"]
    return ens

@pytest.fixture
def sample_data():
    X = pd.DataFrame(
        [[1, 0, 2.1], [0, 1, 1.5]],
        index=["PD_001", "HC_001"],
        columns=["LRRK2_p.G2019S", "GBA_N370S", "SNCA_expr"],
    )
    y = pd.Series({"PD_001": "PD", "HC_001": "HC"})
    mofa = pd.DataFrame({"factor_1": [0.5, -0.2], "factor_2": [-0.3, 0.1]},
                        index=["PD_001", "HC_001"])
    env = pd.Series({"PD_001": 6.8, "HC_001": 2.1})
    stages = pd.Series({"PD_001": "early", "HC_001": None})
    return X, y, mofa, env, stages

def test_build_returns_stage1_outputs(mock_ensemble, sample_data):
    X, y, mofa, env, stages = sample_data
    builder = Stage1Builder(top_k_biomarkers=2)
    outputs = builder.build(X, y, mock_ensemble, mofa, env, stages)
    assert len(outputs) == 2
    assert all(isinstance(o, Stage1Output) for o in outputs)

def test_pd_subject_fields(mock_ensemble, sample_data):
    X, y, mofa, env, stages = sample_data
    builder = Stage1Builder(top_k_biomarkers=2)
    outputs = builder.build(X, y, mock_ensemble, mofa, env, stages)
    pd_out = next(o for o in outputs if o.subject_id == "PD_001")
    assert pd_out.diagnosis == "PD"
    assert pd_out.disease_stage == "early"
    assert abs(pd_out.prediction_confidence - 0.91) < 1e-6
    assert len(pd_out.top_biomarkers) == 2
    assert pd_out.environmental_risk_score == 6.8

def test_hc_subject_no_stage(mock_ensemble, sample_data):
    X, y, mofa, env, stages = sample_data
    builder = Stage1Builder(top_k_biomarkers=2)
    outputs = builder.build(X, y, mock_ensemble, mofa, env, stages)
    hc_out = next(o for o in outputs if o.subject_id == "HC_001")
    assert hc_out.diagnosis == "HC"
    assert hc_out.disease_stage is None

def test_biomarker_shap_direction(mock_ensemble, sample_data):
    X, y, mofa, env, stages = sample_data
    builder = Stage1Builder(top_k_biomarkers=3)
    outputs = builder.build(X, y, mock_ensemble, mofa, env, stages)
    pd_out = next(o for o in outputs if o.subject_id == "PD_001")
    lrrk2 = next(b for b in pd_out.top_biomarkers if b.feature == "LRRK2_p.G2019S")
    assert lrrk2.direction == "up"  # SHAP=0.34 > 0
