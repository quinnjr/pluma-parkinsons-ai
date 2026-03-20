from __future__ import annotations
import pandas as pd
import numpy as np


class TranscriptomicsPreprocessor:
    def __init__(self, min_count: int = 10, min_sample_fraction: float = 0.1):
        self.min_count = min_count
        self.min_sample_fraction = min_sample_fraction

    def normalize_tpm(self, counts: pd.DataFrame,
                      gene_lengths_bp: pd.Series) -> pd.DataFrame:
        """TPM normalization: normalize by gene length (kb), then scale columns to 1e6."""
        lengths_kb = gene_lengths_bp.reindex(counts.index) / 1000.0
        rpk = counts.div(lengths_kb, axis=0)
        scaling = rpk.sum(axis=0) / 1e6
        return rpk.div(scaling, axis=1)

    def filter_low_expression(self, counts: pd.DataFrame) -> pd.DataFrame:
        """Remove genes with fewer than min_count reads in fewer than min_sample_fraction of samples."""
        min_samples = max(1, int(np.ceil(counts.shape[1] * self.min_sample_fraction)))
        expressed = (counts >= self.min_count).sum(axis=1) >= min_samples
        return counts.loc[expressed]

    def log1p_transform(self, matrix: pd.DataFrame) -> pd.DataFrame:
        return np.log1p(matrix)

    def preprocess(self, counts: pd.DataFrame,
                   gene_lengths_bp: pd.Series) -> pd.DataFrame:
        """Full pipeline: filter low-expression -> TPM normalize -> log1p."""
        filtered = self.filter_low_expression(counts)
        tpm = self.normalize_tpm(filtered, gene_lengths_bp)
        return self.log1p_transform(tpm)
