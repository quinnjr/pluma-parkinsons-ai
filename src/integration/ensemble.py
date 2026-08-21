from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


@dataclass
class EnsembleFit:
    """Per-subject predictions and attributions, with their provenance attached.

    ``out_of_fold`` is the field that matters. In-sample SHAP values on a
    boosted-tree model fitted to the same rows are close to a restatement of the
    labels; reporting them as biomarker evidence overstates the finding, so the
    flag travels with the numbers all the way into the generated text.
    """

    proba: np.ndarray
    shap_values: np.ndarray
    feature_names: list[str]
    out_of_fold: bool
    cv_auc: float | None
    n_splits: int

    def __post_init__(self) -> None:
        if self.shap_values.shape[0] != self.proba.shape[0]:
            raise ValueError(
                f"shap_values has {self.shap_values.shape[0]} rows but proba has "
                f"{self.proba.shape[0]}"
            )
        if self.shap_values.shape[1] != len(self.feature_names):
            raise ValueError(
                f"shap_values has {self.shap_values.shape[1]} columns but there are "
                f"{len(self.feature_names)} feature names"
            )


class OmicsEnsemble:
    """XGBoost classifier with SHAP feature attribution for multi-omics data."""

    LABEL_MAP = {"PD": 1, "HC": 0, "SWEDD": 0, "Prodromal": 0}

    def __init__(self, n_estimators: int = 200, max_depth: int = 6,
                 learning_rate: float = 0.05, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = self._new_model()
        self._feature_names: list[str] = []
        self._shap_values: np.ndarray | None = None

    def _new_model(self) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            eval_metric="logloss",
        )

    def encode_labels(self, y: pd.Series) -> pd.Series:
        return y.map(self.LABEL_MAP).fillna(0).astype(int)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._feature_names = [str(c) for c in X.columns]
        self.model.fit(X.values, self.encode_labels(y).values)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X.values)[:, 1]

    def compute_shap(self, X: pd.DataFrame) -> np.ndarray:
        self._shap_values = shap.TreeExplainer(self.model).shap_values(X.values)
        return self._shap_values

    def top_features(self, n: int = 20) -> list[tuple[str, float]]:
        importance = self.model.feature_importances_
        ranked = sorted(zip(self._feature_names, importance.tolist()),
                        key=lambda pair: pair[1], reverse=True)
        return ranked[:n]

    # -- evaluation ------------------------------------------------------------

    def usable_splits(self, y: pd.Series, requested: int) -> int:
        """Largest workable number of stratified folds, or 0 if CV is impossible."""
        counts = self.encode_labels(y).value_counts()
        if len(counts) < 2:
            return 0
        smallest_class = int(counts.min())
        return min(requested, smallest_class) if smallest_class >= 2 else 0

    def fit_evaluate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> EnsembleFit:
        """Fit on all rows, but return out-of-fold predictions and attributions.

        Falls back to in-sample values when the cohort cannot be stratified into
        at least two folds (fewer than two subjects in the minority class). The
        fallback is flagged, never silent.
        """
        self._feature_names = [str(c) for c in X.columns]
        y_binary = self.encode_labels(y)
        splits = self.usable_splits(y, n_splits)

        # The full-data model is what callers reuse for `top_features` and for
        # scoring new subjects; the fold models exist only to produce honest
        # per-subject numbers.
        self.fit(X, y)

        if splits < 2:
            logger.warning(
                "Cohort cannot be split into >=2 stratified folds (class counts: %s). "
                "Falling back to in-sample predictions and SHAP; feature importance will "
                "be optimistic.",
                y_binary.value_counts().to_dict(),
            )
            return EnsembleFit(
                proba=self.predict_proba(X),
                shap_values=np.asarray(self.compute_shap(X)),
                feature_names=list(self._feature_names),
                out_of_fold=False,
                cv_auc=None,
                n_splits=0,
            )

        oof_proba = np.zeros(len(X), dtype=float)
        oof_shap = np.zeros((len(X), X.shape[1]), dtype=float)
        folds = StratifiedKFold(n_splits=splits, shuffle=True, random_state=self.random_state)
        for train_idx, test_idx in folds.split(X.values, y_binary.values):
            fold_model = self._new_model()
            fold_model.fit(X.values[train_idx], y_binary.values[train_idx])
            held_out = X.values[test_idx]
            oof_proba[test_idx] = fold_model.predict_proba(held_out)[:, 1]
            oof_shap[test_idx] = np.asarray(shap.TreeExplainer(fold_model).shap_values(held_out))

        self._shap_values = oof_shap
        return EnsembleFit(
            proba=oof_proba,
            shap_values=oof_shap,
            feature_names=list(self._feature_names),
            out_of_fold=True,
            cv_auc=float(roc_auc_score(y_binary.values, oof_proba)),
            n_splits=splits,
        )
