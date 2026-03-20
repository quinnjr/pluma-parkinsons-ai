# tests/integration/test_integration.py
import pytest
import pandas as pd
import numpy as np
from src.integration.ensemble import OmicsEnsemble

@pytest.fixture
def feature_matrix():
    np.random.seed(1)
    n = 60
    subjects = [f"S{i}" for i in range(n)]
    X = pd.DataFrame(np.random.randn(n, 50), index=subjects)
    y = pd.Series(["PD"] * 30 + ["HC"] * 30, index=subjects)
    return X, y

def test_ensemble_fit_predict(feature_matrix):
    X, y = feature_matrix
    ensemble = OmicsEnsemble(n_estimators=10)
    ensemble.fit(X, y)
    preds = ensemble.predict_proba(X)
    assert len(preds) == len(y)
    assert all(0.0 <= p <= 1.0 for p in preds)

def test_ensemble_top_features(feature_matrix):
    X, y = feature_matrix
    ensemble = OmicsEnsemble(n_estimators=10)
    ensemble.fit(X, y)
    top = ensemble.top_features(n=5)
    assert len(top) == 5
    assert all(isinstance(name, str) for name, _ in top)
    assert all(score >= 0 for _, score in top)
    # Features should be sorted by importance descending
    scores = [s for _, s in top]
    assert scores == sorted(scores, reverse=True)

def test_ensemble_label_map(feature_matrix):
    X, y = feature_matrix
    ensemble = OmicsEnsemble(n_estimators=10)
    ensemble.fit(X, y)
    # PD = 1, HC = 0 in LABEL_MAP
    assert ensemble.LABEL_MAP["PD"] == 1
    assert ensemble.LABEL_MAP["HC"] == 0
