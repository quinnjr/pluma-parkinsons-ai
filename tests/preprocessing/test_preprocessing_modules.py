# tests/preprocessing/test_preprocessing_modules.py
import pytest
import pandas as pd
import numpy as np
from src.preprocessing.epigenomics import EpigenomicsPreprocessor
from src.preprocessing.proteomics import ProteomicsPreprocessor
from src.preprocessing.metabolomics import MetabolomicsPreprocessor
from src.preprocessing.microbiome import MicrobiomePreprocessor
from src.preprocessing.environmental_prep import EnvironmentalPreprocessor
from src.preprocessing.clinical import ClinicalPreprocessor

def test_epigenomics_beta_clamp():
    proc = EpigenomicsPreprocessor()
    df = pd.DataFrame({"CpG_1": [-0.1, 0.0, 0.5, 1.0, 1.1]},
                      index=[f"S{i}" for i in range(5)])
    clamped = proc.clamp_beta(df)
    assert clamped.min().min() >= 0.0
    assert clamped.max().max() <= 1.0

def test_epigenomics_m_value_transform():
    proc = EpigenomicsPreprocessor()
    df = pd.DataFrame({"CpG_1": [0.1, 0.5, 0.9]},
                      index=["S0", "S1", "S2"])
    m = proc.m_value_transform(df)
    # M-value of 0.5 beta = log2(0.5/0.5) = 0
    assert abs(m["CpG_1"].iloc[1]) < 1e-9

def test_proteomics_log2_transform():
    proc = ProteomicsPreprocessor()
    df = pd.DataFrame({"prot_A": [1.0, 4.0, 8.0], "prot_B": [2.0, 2.0, 2.0]})
    transformed = proc.log2_transform(df)
    assert abs(transformed["prot_A"].iloc[1] - 2.0) < 1e-9  # log2(4) == 2

def test_proteomics_impute_missing():
    proc = ProteomicsPreprocessor(n_neighbors=2)
    df = pd.DataFrame({"prot_A": [1.0, np.nan, 3.0, 4.0],
                       "prot_B": [2.0, 2.0, 2.0, 2.0]})
    imputed = proc.impute_missing(df)
    assert not imputed.isnull().any().any()

def test_metabolomics_clr_row_sums_zero():
    proc = MetabolomicsPreprocessor()
    df = pd.DataFrame({"m1": [1.0, 2.0], "m2": [3.0, 4.0], "m3": [2.0, 1.0]})
    clr = proc.clr_transform(df)
    assert (clr.sum(axis=1).abs() < 1e-9).all()

def test_microbiome_clr_shape_preserved():
    proc = MicrobiomePreprocessor(rarefaction_depth=100)
    df = pd.DataFrame({"bug_A": [50.0, 30.0], "bug_B": [50.0, 70.0]},
                      index=["S0", "S1"])
    clr = proc.clr_transform(df)
    assert clr.shape == df.shape

def test_environmental_prep_fills_na():
    proc = EnvironmentalPreprocessor()
    df = pd.DataFrame({"pm25": [5.0, np.nan, 7.0], "no2": [10.0, 12.0, np.nan]})
    result = proc.preprocess(df)
    assert not result.isnull().any().any()

def test_clinical_updrs_total():
    proc = ClinicalPreprocessor()
    df = pd.DataFrame({
        "subject_id": ["PD_001"],
        "UPDRS_I": [5], "UPDRS_II": [10], "UPDRS_III": [20], "UPDRS_IV": [3],
        "HOEHN_YAHR": [2.0],
    })
    processed = proc.process(df)
    assert processed["updrs_total"].iloc[0] == 38
    assert "disease_stage" in processed.columns
    assert processed["disease_stage"].iloc[0] == "early"

def test_clinical_hc_no_stage():
    proc = ClinicalPreprocessor()
    df = pd.DataFrame({
        "subject_id": ["HC_001"],
        "UPDRS_I": [0], "UPDRS_II": [0], "UPDRS_III": [0], "UPDRS_IV": [0],
        "HOEHN_YAHR": [0.0],
    })
    processed = proc.process(df)
    assert processed["disease_stage"].iloc[0] is None
