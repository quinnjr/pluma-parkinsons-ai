from __future__ import annotations
import pandas as pd

HY_STAGE_MAP: dict = {
    0.0: None, 1.0: "early", 1.5: "early",
    2.0: "early", 2.5: "early", 3.0: "mid",
    4.0: "late", 5.0: "late",
}


class ClinicalPreprocessor:
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        updrs_cols = [c for c in ["UPDRS_I", "UPDRS_II", "UPDRS_III", "UPDRS_IV"]
                      if c in df.columns]
        if updrs_cols:
            df["updrs_total"] = df[updrs_cols].sum(axis=1)
        if "HOEHN_YAHR" in df.columns:
            df["disease_stage"] = df["HOEHN_YAHR"].map(HY_STAGE_MAP)
        return df
