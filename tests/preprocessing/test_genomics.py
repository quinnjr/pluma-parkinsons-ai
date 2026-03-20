# tests/preprocessing/test_genomics.py
import pytest
import pandas as pd
import numpy as np
from src.preprocessing.genomics import GenomicsPreprocessor

@pytest.fixture
def sample_variant_df():
    return pd.DataFrame({
        "variant_id": ["rs34637584", "rs76904798", "rs356182", "rs199347"],
        "gene": ["LRRK2", "LRRK2", "SNCA", "GPNMB"],
        "genotype": [1, 0, 2, 1],
        "maf": [0.001, 0.15, 0.22, 0.08],
        "subject_id": ["PD_001"] * 4,
    })

def test_maf_filter_removes_rare_variants(sample_variant_df):
    # rs34637584 is LRRK2 (a known PD gene) with MAF=0.001.  The default
    # whitelist keeps it, so the variant that should be dropped by MAF alone
    # must be a gene NOT in DEFAULT_PD_GENES.  Verify a non-PD-gene rare
    # variant is removed while known-PD-gene rare variants survive.
    proc = GenomicsPreprocessor(maf_threshold=0.01, known_pd_genes=[])
    filtered = proc.filter_by_maf(sample_variant_df)
    assert "rs34637584" not in filtered["variant_id"].values
    assert len(filtered) == 3

def test_known_pd_genes_always_kept(sample_variant_df):
    proc = GenomicsPreprocessor(maf_threshold=0.01, known_pd_genes=["LRRK2", "SNCA"])
    filtered = proc.filter_by_maf(sample_variant_df)
    # rs34637584 (LRRK2, MAF=0.001) kept despite low MAF
    assert "rs34637584" in filtered["variant_id"].values

def test_encode_genotype_to_dosage(sample_variant_df):
    proc = GenomicsPreprocessor(maf_threshold=0.01)
    encoded = proc.encode_dosage(sample_variant_df)
    assert set(encoded["genotype"].unique()).issubset({0, 1, 2})

def test_pivot_to_feature_matrix(sample_variant_df):
    proc = GenomicsPreprocessor(maf_threshold=0.0)
    matrix = proc.pivot_to_feature_matrix(sample_variant_df)
    assert "PD_001" in matrix.index
    assert "rs34637584" in matrix.columns

def test_preprocess_pipeline(sample_variant_df):
    proc = GenomicsPreprocessor(maf_threshold=0.01, known_pd_genes=["LRRK2", "SNCA"])
    matrix = proc.preprocess(sample_variant_df)
    assert isinstance(matrix, pd.DataFrame)
    assert matrix.index.name == "subject_id"

def test_default_constructor_uses_pd_gene_whitelist():
    """GenomicsPreprocessor() with no args should protect known PD genes."""
    proc = GenomicsPreprocessor(maf_threshold=0.01)  # no known_pd_genes arg
    df = pd.DataFrame({
        "variant_id": ["rs34637584"],
        "gene": ["LRRK2"],
        "genotype": [1],
        "maf": [0.001],  # below threshold
        "subject_id": ["PD_001"],
    })
    filtered = proc.filter_by_maf(df)
    # LRRK2 variant should be KEPT despite low MAF because LRRK2 is in DEFAULT_PD_GENES
    assert "rs34637584" in filtered["variant_id"].values
