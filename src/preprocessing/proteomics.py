from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


class ProteomicsPreprocessor:
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors

    def log2_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return np.log2(df.clip(lower=1e-9))

    def impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        imputer = KNNImputer(n_neighbors=self.n_neighbors)
        return pd.DataFrame(imputer.fit_transform(df),
                            index=df.index, columns=df.columns)

    def zscore_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        return (df - df.mean()) / (df.std() + 1e-9)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.zscore_normalize(self.impute_missing(self.log2_transform(df)))
