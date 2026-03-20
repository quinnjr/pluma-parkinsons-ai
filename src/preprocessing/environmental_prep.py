from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class EnvironmentalPreprocessor:
    def compute_air_quality_index(self, df: pd.DataFrame,
                                   metrics: list[str]) -> pd.Series:
        scaler = MinMaxScaler()
        scaled = pd.DataFrame(scaler.fit_transform(df[metrics]),
                               index=df.index, columns=metrics)
        return scaled.mean(axis=1).rename("air_quality_index")

    def compute_exposure_score(self, df: pd.DataFrame,
                                exposure_cols: list[str]) -> pd.Series:
        scaler = MinMaxScaler()
        scaled = pd.DataFrame(scaler.fit_transform(df[exposure_cols]),
                               index=df.index, columns=exposure_cols)
        return (scaled * 10).mean(axis=1).rename("environmental_risk_score")

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.fillna(df.median(numeric_only=True))
