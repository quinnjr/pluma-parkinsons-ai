from __future__ import annotations
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import shap


class OmicsEnsemble:
    """XGBoost classifier with SHAP feature importance for multi-omics data."""

    LABEL_MAP = {"PD": 1, "HC": 0, "SWEDD": 0, "Prodromal": 0}

    def __init__(self, n_estimators: int = 200, max_depth: int = 6,
                 learning_rate: float = 0.05, random_state: int = 42):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            eval_metric="logloss",
        )
        self._feature_names: list[str] = []
        self._shap_values: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._feature_names = [str(c) for c in X.columns]
        y_binary = y.map(self.LABEL_MAP).fillna(0).astype(int)
        self.model.fit(X.values, y_binary.values)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X.values)[:, 1]

    def compute_shap(self, X: pd.DataFrame) -> np.ndarray:
        explainer = shap.TreeExplainer(self.model)
        self._shap_values = explainer.shap_values(X.values)
        return self._shap_values

    def top_features(self, n: int = 20) -> list[tuple[str, float]]:
        importance = self.model.feature_importances_
        ranked = sorted(zip(self._feature_names, importance.tolist()),
                        key=lambda x: x[1], reverse=True)
        return ranked[:n]
