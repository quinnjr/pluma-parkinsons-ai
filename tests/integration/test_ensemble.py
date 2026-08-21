import numpy as np
import pandas as pd
import pytest

from src.integration.ensemble import EnsembleFit, OmicsEnsemble


def _cohort(n_per_class: int = 20, n_features: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    X_pd = rng.normal(1.0, 1.0, size=(n_per_class, n_features))
    X_hc = rng.normal(-1.0, 1.0, size=(n_per_class, n_features))
    X = pd.DataFrame(
        np.vstack([X_pd, X_hc]),
        index=[f"PD_{i}" for i in range(n_per_class)]
        + [f"HC_{i}" for i in range(n_per_class)],
        columns=[f"feat_{j}" for j in range(n_features)],
    )
    y = pd.Series(["PD"] * n_per_class + ["HC"] * n_per_class, index=X.index)
    return X, y


def test_ensemble_fit_shape_validation():
    with pytest.raises(ValueError):
        EnsembleFit(proba=np.zeros(2), shap_values=np.zeros((3, 2)),
                    feature_names=["a", "b"], out_of_fold=True, cv_auc=None, n_splits=2)
    with pytest.raises(ValueError):
        EnsembleFit(proba=np.zeros(2), shap_values=np.zeros((2, 3)),
                    feature_names=["a", "b"], out_of_fold=True, cv_auc=None, n_splits=2)


def test_encode_labels_maps_non_pd_to_zero():
    ens = OmicsEnsemble()
    y = pd.Series(["PD", "HC", "SWEDD", "Prodromal", "???"])
    assert ens.encode_labels(y).tolist() == [1, 0, 0, 0, 0]


def test_usable_splits():
    ens = OmicsEnsemble()
    assert ens.usable_splits(pd.Series(["PD"] * 10 + ["HC"] * 10), 5) == 5
    assert ens.usable_splits(pd.Series(["PD"] * 3 + ["HC"] * 10), 5) == 3
    assert ens.usable_splits(pd.Series(["PD", "HC", "HC"]), 5) == 0  # minority of 1
    assert ens.usable_splits(pd.Series(["PD", "PD", "PD"]), 5) == 0  # one class


def test_fit_evaluate_is_out_of_fold_on_a_real_cohort():
    X, y = _cohort()
    fit = OmicsEnsemble(n_estimators=20).fit_evaluate(X, y, n_splits=4)
    assert fit.out_of_fold is True
    assert fit.n_splits == 4
    assert fit.proba.shape == (40,)
    assert fit.shap_values.shape == (40, 5)
    assert fit.feature_names == list(X.columns)
    # Strongly separated classes: out-of-fold AUC should be excellent, and it
    # must be an estimate, not the in-sample 1.0-by-construction.
    assert fit.cv_auc is not None
    assert 0.8 < fit.cv_auc <= 1.0


def test_fit_evaluate_falls_back_flagged_on_tiny_cohort():
    X, y = _cohort(n_per_class=1)
    fit = OmicsEnsemble(n_estimators=5).fit_evaluate(X, y)
    assert fit.out_of_fold is False
    assert fit.cv_auc is None
    assert fit.n_splits == 0
    assert fit.proba.shape == (2,)


def test_fit_evaluate_keeps_full_model_usable():
    X, y = _cohort()
    ens = OmicsEnsemble(n_estimators=20)
    ens.fit_evaluate(X, y)
    assert ens.predict_proba(X.iloc[:3]).shape == (3,)
    top = ens.top_features(3)
    assert len(top) == 3
    assert all(name in X.columns for name, _ in top)
