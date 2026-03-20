# tests/integration/test_mofa.py
import pytest
import pandas as pd
import numpy as np
from src.integration.mofa import MOFAIntegrator

@pytest.fixture
def small_omics():
    np.random.seed(0)
    n = 30
    subjects = [f"S{i}" for i in range(n)]
    return {
        "genomics": pd.DataFrame(np.random.randn(n, 20), index=subjects),
        "transcriptomics": pd.DataFrame(np.random.randn(n, 30), index=subjects),
        "proteomics": pd.DataFrame(np.random.randn(n, 15), index=subjects),
    }

def test_mofa_initializes():
    integrator = MOFAIntegrator(n_factors=5)
    assert integrator.n_factors == 5

def test_align_subjects_keeps_common_only(small_omics):
    integrator = MOFAIntegrator(n_factors=5)
    small_omics["proteomics"] = small_omics["proteomics"].iloc[:-3]
    aligned = integrator.align_subjects(small_omics)
    n_common = len(aligned["genomics"])
    for df in aligned.values():
        assert len(df) == n_common
    assert n_common == 27

def test_align_subjects_sorts_index(small_omics):
    integrator = MOFAIntegrator(n_factors=5)
    aligned = integrator.align_subjects(small_omics)
    for df in aligned.values():
        assert list(df.index) == sorted(df.index)
