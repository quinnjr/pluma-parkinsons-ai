# tests/preprocessing/test_transcriptomics.py
import pytest
import pandas as pd
import numpy as np
from src.preprocessing.transcriptomics import TranscriptomicsPreprocessor

@pytest.fixture
def raw_counts():
    np.random.seed(42)
    data = np.random.randint(10, 1000, size=(100, 20))
    genes = [f"GENE_{i}" for i in range(100)]
    samples = [f"S{i}" for i in range(20)]
    return pd.DataFrame(data, index=genes, columns=samples)

@pytest.fixture
def gene_lengths(raw_counts):
    np.random.seed(0)
    return pd.Series(
        np.random.randint(500, 5000, size=len(raw_counts.index)),
        index=raw_counts.index,
    )

def test_tpm_columns_sum_to_1e6(raw_counts, gene_lengths):
    proc = TranscriptomicsPreprocessor()
    tpm = proc.normalize_tpm(raw_counts, gene_lengths)
    col_sums = tpm.sum(axis=0)
    assert (np.abs(col_sums - 1e6) < 1).all(), "Each TPM column must sum to 1e6"

def test_low_expression_filter_removes_genes(raw_counts):
    # Set first 10 genes to all-zeros (should be filtered)
    raw_counts.iloc[:10, :] = 0
    proc = TranscriptomicsPreprocessor(min_count=10, min_sample_fraction=0.1)
    filtered = proc.filter_low_expression(raw_counts)
    assert filtered.shape[0] < raw_counts.shape[0]
    assert filtered.shape[0] == 90

def test_log1p_transform_increases_not_decreases(raw_counts):
    proc = TranscriptomicsPreprocessor()
    transformed = proc.log1p_transform(raw_counts)
    assert (transformed >= 0).all().all()
    # log1p(x) < x for x > 0
    assert transformed.max().max() < raw_counts.max().max()

def test_preprocess_pipeline(raw_counts, gene_lengths):
    proc = TranscriptomicsPreprocessor()
    result = proc.preprocess(raw_counts, gene_lengths)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[1] == raw_counts.shape[1]
