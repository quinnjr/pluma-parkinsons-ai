from __future__ import annotations

import numpy as np
import pandas as pd


class MetabolomicsPreprocessor:
    def clr_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Centered log-ratio transform for compositional metabolomics data."""
        df_pos = df.clip(lower=1e-9)
        log_df = np.log(df_pos)
        log_gm = log_df.mean(axis=1)
        return log_df.subtract(log_gm, axis=0)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.clr_transform(df)
