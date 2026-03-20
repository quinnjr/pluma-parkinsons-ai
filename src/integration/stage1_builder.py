from __future__ import annotations
import pandas as pd
import numpy as np
from src.models import Stage1Output, BiomarkerHit


class Stage1Builder:
    """Assembles Stage1Output objects from ensemble predictions, SHAP values, and MOFA factors."""

    def __init__(self, top_k_biomarkers: int = 20):
        self.top_k_biomarkers = top_k_biomarkers

    def build(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        ensemble,
        mofa_factors: pd.DataFrame,
        environmental_scores: pd.Series,
        disease_stages: pd.Series,
    ) -> list[Stage1Output]:
        probas = ensemble.predict_proba(X)
        shap_values = ensemble.compute_shap(X)
        feature_names = ensemble._feature_names
        outputs = []

        for i, subject_id in enumerate(X.index):
            shap_row = shap_values[i]
            top_idx = np.argsort(np.abs(shap_row))[::-1][: self.top_k_biomarkers]
            top_biomarkers = [
                BiomarkerHit(
                    modality=self._infer_modality(feature_names[j]),
                    feature=feature_names[j],
                    shap_value=float(shap_row[j]),
                    direction="up" if shap_row[j] > 0 else "down",
                )
                for j in top_idx
            ]
            factors = (
                mofa_factors.loc[subject_id].to_dict()
                if subject_id in mofa_factors.index
                else {}
            )
            stage = disease_stages.get(subject_id) if disease_stages is not None else None
            outputs.append(
                Stage1Output(
                    subject_id=subject_id,
                    diagnosis=y[subject_id],
                    prediction_confidence=float(probas[i]),
                    disease_stage=stage,
                    top_biomarkers=top_biomarkers,
                    mofa_factors=factors,
                    environmental_risk_score=float(
                        environmental_scores.get(subject_id, 0.0)
                    ),
                )
            )
        return outputs

    def _infer_modality(self, feature_name: str) -> str:
        """Heuristic modality assignment based on feature naming conventions."""
        name = feature_name.lower()
        if any(g in name for g in ["rs", "lrrk2", "gba", "snca", "snp", "variant", "_p."]):
            return "genomics"
        if "_expr" in name or "rna" in name:
            return "transcriptomics"
        if "cpg" in name or "methylation" in name:
            return "epigenomics"
        if "prot" in name or "csf_" in name:
            return "proteomics"
        if "metab" in name or "metabolite" in name:
            return "metabolomics"
        if "bug_" in name or "bacteria" in name:
            return "microbiome"
        if any(e in name for e in ["pm25", "pm10", "pesticide", "metal", "air"]):
            return "environmental"
        return "clinical"
