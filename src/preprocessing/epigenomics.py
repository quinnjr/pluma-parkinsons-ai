from __future__ import annotations
import pandas as pd
import numpy as np


class EpigenomicsPreprocessor:
    def clamp_beta(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.clip(0.0, 1.0)

    def m_value_transform(self, beta: pd.DataFrame) -> pd.DataFrame:
        """Convert beta values to M-values: log2(beta / (1 - beta))."""
        beta_clamped = beta.clip(1e-6, 1 - 1e-6)
        return np.log2(beta_clamped / (1 - beta_clamped))

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.m_value_transform(self.clamp_beta(df))
