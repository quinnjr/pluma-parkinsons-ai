import numpy as np
import pandas as pd
import pytest

from src.integration.ensemble import EnsembleFit
from src.integration.stage1_builder import Stage1Builder
from src.models import Provenance, Stage1Output

FEATURES = ["LRRK2_p.G2019S", "GBA_N370S", "SNCA_expr"]


@pytest.fixture
def fit():
    return EnsembleFit(
        proba=np.array([0.91, 0.08]),
        shap_values=np.array([[0.34, -0.10, 0.21],
                              [-0.05, 0.02, -0.01]]),
        feature_names=FEATURES,
        out_of_fold=True,
        cv_auc=0.88,
        n_splits=2,
    )


@pytest.fixture
def sample_data():
    X = pd.DataFrame(
        [[1, 0, 2.1], [0, 1, 1.5]],
        index=["PD_001", "HC_001"],
        columns=FEATURES,
    )
    y = pd.Series({"PD_001": "PD", "HC_001": "HC"})
    mofa = pd.DataFrame({"factor_1": [0.5, -0.2], "factor_2": [-0.3, 0.1]},
                        index=["PD_001", "HC_001"])
    env = pd.Series({"PD_001": 6.8, "HC_001": 2.1})
    stages = pd.Series({"PD_001": "early", "HC_001": None})
    return X, y, mofa, env, stages


def test_build_returns_stage1_outputs(fit, sample_data):
    outputs = Stage1Builder(top_k_biomarkers=2).build(*sample_data[:2], fit,
                                                      *sample_data[2:])
    assert len(outputs) == 2
    assert all(isinstance(o, Stage1Output) for o in outputs)


def test_pd_subject_fields(fit, sample_data):
    outputs = Stage1Builder(top_k_biomarkers=2).build(*sample_data[:2], fit,
                                                      *sample_data[2:])
    pd_out = next(o for o in outputs if o.subject_id == "PD_001")
    assert pd_out.diagnosis == "PD"
    assert pd_out.disease_stage == "early"
    assert pd_out.prediction_confidence == pytest.approx(0.91)
    assert len(pd_out.top_biomarkers) == 2
    assert pd_out.environmental_risk_score == 6.8
    assert pd_out.mofa_factors == {"factor_1": 0.5, "factor_2": -0.3}


def test_hc_subject_no_stage(fit, sample_data):
    outputs = Stage1Builder(top_k_biomarkers=2).build(*sample_data[:2], fit,
                                                      *sample_data[2:])
    hc_out = next(o for o in outputs if o.subject_id == "HC_001")
    assert hc_out.diagnosis == "HC"
    assert hc_out.disease_stage is None


def test_biomarker_effect_follows_shap_sign(fit, sample_data):
    outputs = Stage1Builder(top_k_biomarkers=3).build(*sample_data[:2], fit,
                                                      *sample_data[2:])
    pd_out = next(o for o in outputs if o.subject_id == "PD_001")
    by_feature = {b.feature: b for b in pd_out.top_biomarkers}
    assert by_feature["LRRK2_p.G2019S"].effect == "toward_pd"  # SHAP +0.34
    assert by_feature["GBA_N370S"].effect == "away_from_pd"    # SHAP -0.10


def test_biomarkers_ranked_by_abs_shap(fit, sample_data):
    outputs = Stage1Builder(top_k_biomarkers=3).build(*sample_data[:2], fit,
                                                      *sample_data[2:])
    pd_out = next(o for o in outputs if o.subject_id == "PD_001")
    shap_magnitudes = [abs(b.shap_value) for b in pd_out.top_biomarkers]
    assert shap_magnitudes == sorted(shap_magnitudes, reverse=True)


def test_value_z_is_cohort_standardised(fit, sample_data):
    outputs = Stage1Builder(top_k_biomarkers=3).build(*sample_data[:2], fit,
                                                      *sample_data[2:])
    pd_out = next(o for o in outputs if o.subject_id == "PD_001")
    lrrk2 = next(b for b in pd_out.top_biomarkers if b.feature == "LRRK2_p.G2019S")
    # Values [1, 0], population std 0.5 -> z = (1 - 0.5) / 0.5 = +1.
    assert lrrk2.value_z == pytest.approx(1.0)


def test_provenance_reflects_fit(fit, sample_data):
    supplied = Provenance(synthetic_modalities=("microbiome",), datasets=("GSE6613",))
    outputs = Stage1Builder(top_k_biomarkers=2).build(*sample_data[:2], fit,
                                                      *sample_data[2:],
                                                      provenance=supplied)
    prov = outputs[0].provenance
    assert prov.shap_out_of_fold is True
    assert prov.cv_auc == pytest.approx(0.88)
    assert prov.cohort_size == 2
    assert prov.synthetic_modalities == ("microbiome",)
    assert prov.datasets == ("GSE6613",)


def test_row_mismatch_raises(fit, sample_data):
    X, y, mofa, env, stages = sample_data
    with pytest.raises(ValueError, match="rows"):
        Stage1Builder().build(X.iloc[:1], y, fit, mofa, env, stages)


@pytest.mark.parametrize("feature,expected", [
    ("SNCA_expr", "transcriptomics"),
    ("1552281_at", "transcriptomics"),
    ("cg_SNCA", "epigenomics"),
    ("csf_NfL", "proteomics"),
    ("metab_urate", "metabolomics"),
    ("bug_Prevotella", "microbiome"),
    ("g__Akkermansia", "microbiome"),
    ("pm25_annual", "environmental"),
    ("rs34637584", "genomics"),
    ("LRRK2", "genomics"),
    ("factor_3", "integrated"),
    ("updrs_iii", "clinical"),
    ("microbiome:mystery_taxon", "microbiome"),
])
def test_infer_modality(feature, expected):
    assert Stage1Builder.infer_modality(feature) == expected


def test_nan_disease_stage_becomes_none(fit, sample_data):
    X, y, mofa, env, _ = sample_data
    # A stage Series with no strings is float-dtyped: missing values are NaN.
    stages = pd.Series([np.nan, np.nan], index=["PD_001", "HC_001"])
    outputs = Stage1Builder(top_k_biomarkers=2).build(X, y, fit, mofa, env, stages)
    assert all(o.disease_stage is None for o in outputs)


def test_zero_shap_features_are_not_biomarkers(sample_data):
    X, y, mofa, env, stages = sample_data
    fit = EnsembleFit(
        proba=np.array([0.91, 0.08]),
        shap_values=np.array([[0.34, 0.0, 0.21], [0.0, 0.0, 0.0]]),
        feature_names=FEATURES,
        out_of_fold=True,
        cv_auc=0.88,
        n_splits=2,
    )
    outputs = Stage1Builder(top_k_biomarkers=3).build(X, y, fit, mofa, env, stages)
    pd_out = next(o for o in outputs if o.subject_id == "PD_001")
    assert [b.feature for b in pd_out.top_biomarkers] == ["LRRK2_p.G2019S", "SNCA_expr"]
    hc_out = next(o for o in outputs if o.subject_id == "HC_001")
    assert hc_out.top_biomarkers == []


def test_integrated_prefix_maps_to_integrated_modality():
    assert Stage1Builder.infer_modality("integrated:factor_1") == "integrated"
    assert Stage1Builder.infer_modality("integrated:factor_12") == "integrated"


def test_namespaced_prefix_wins_over_stem_patterns():
    # The prefix is the pipeline's authoritative label; only the remainder is
    # scanned when the prefix itself is not a known modality.
    assert Stage1Builder.infer_modality("transcriptomics:NEFL") == "transcriptomics"
    assert Stage1Builder.infer_modality("unknownprefix:SNCA_expr") == "transcriptomics"


def test_missing_environmental_score_raises(fit, sample_data):
    X, y, mofa, env, stages = sample_data
    env_missing = env.drop("HC_001")
    with pytest.raises(ValueError, match="environmental risk score"):
        Stage1Builder(top_k_biomarkers=2).build(X, y, fit, mofa, env_missing, stages)
