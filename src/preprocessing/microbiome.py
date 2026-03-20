from __future__ import annotations
import pandas as pd
import numpy as np


class MicrobiomePreprocessor:
    def __init__(self, rarefaction_depth: int = 10000):
        self.rarefaction_depth = rarefaction_depth

    def rarefy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Subsample each sample to rarefaction_depth reads."""
        def _rarefy_row(row: pd.Series) -> pd.Series:
            total = row.sum()
            if total <= self.rarefaction_depth:
                return row
            proportions = row / total
            resampled = np.random.multinomial(self.rarefaction_depth, proportions)
            return pd.Series(resampled.astype(float), index=row.index)
        return df.apply(_rarefy_row, axis=1)

    def clr_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_pos = df.clip(lower=1e-9)
        log_df = np.log(df_pos)
        return log_df.subtract(log_df.mean(axis=1), axis=0)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.clr_transform(self.rarefy(df))
