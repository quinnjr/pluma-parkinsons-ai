from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.integration.ensemble import EnsembleFit
from src.models import AWAY_FROM_PD, TOWARD_PD, BiomarkerHit, Provenance, Stage1Output

#: Feature-name fragments that identify a modality, checked in order: the first
#: group whose fragment appears in the name wins.
MODALITY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Measurement-type fragments first: "SNCA_expr" is transcriptomics, not
    # genomics, and only the suffix says so.
    ("transcriptomics", ("expr", "rna", "transcript", "_at", "tpm", "fpkm")),
    ("epigenomics", ("cpg", "cg_", "methyl", "beta_value")),
    ("proteomics", ("prot", "csf_", "nfl", "peptide")),
    ("metabolomics", ("metab", "metabolite", "urate", "caffeine", "hmdb")),
    ("microbiome", ("bug_", "otu", "asv", "g__", "f__", "bacteri", "prevotella", "akkermansia")),
    ("environmental", ("pm25", "pm10", "pesticide", "paraquat", "rotenone", "lead", "cadmium",
                       "metal", "air_", "lbxb")),
    # Bare gene symbols fall through to genomics last.
    ("genomics", ("snp", "variant", "allele", "genotype", "lrrk2", "gba", "snca", "prkn",
                  "pink1", "park7", "vps35", "mapt")),
)

#: dbSNP identifiers are unambiguous, unlike the bare substring "rs".
RSID_PATTERN = re.compile(r"(?:^|[^a-z0-9])rs\d+")


class Stage1Builder:
    """Assemble per-subject `Stage1Output` records from an `EnsembleFit`."""

    def __init__(self, top_k_biomarkers: int = 20):
        self.top_k_biomarkers = top_k_biomarkers

    def build(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fit: EnsembleFit,
        mofa_factors: pd.DataFrame,
        environmental_scores: pd.Series,
        disease_stages: pd.Series | None,
        provenance: Provenance | None = None,
    ) -> list[Stage1Output]:
        if len(X) != len(fit.proba):
            raise ValueError(
                f"X has {len(X)} rows but the fit covers {len(fit.proba)} subjects"
            )
        z_scores = self._z_scores(X)
        base_provenance = self._provenance(X, fit, provenance)
        # The feature panel is fixed across subjects; classify each name once.
        modality_of = {name: self.infer_modality(name) for name in fit.feature_names}

        outputs = []
        for i, subject_id in enumerate(X.index):
            shap_row = fit.shap_values[i]
            # A feature with an attribution of exactly 0 pushed the classifier
            # nowhere; reporting a direction for it would be an invention.
            ranked = [j for j in np.argsort(np.abs(shap_row))[::-1] if shap_row[j] != 0]
            top_idx = ranked[: self.top_k_biomarkers]
            top_biomarkers = [
                BiomarkerHit(
                    modality=modality_of[fit.feature_names[j]],
                    feature=fit.feature_names[j],
                    shap_value=float(shap_row[j]),
                    effect=TOWARD_PD if shap_row[j] > 0 else AWAY_FROM_PD,
                    value_z=self._z_for(z_scores, subject_id, fit.feature_names[j]),
                )
                for j in top_idx
            ]
            factors = (
                mofa_factors.loc[subject_id].to_dict()
                if mofa_factors is not None and subject_id in mofa_factors.index
                else {}
            )
            stage = disease_stages.get(subject_id) if disease_stages is not None else None
            if not isinstance(stage, str) and pd.isna(stage):
                # A Series with no stages is float-dtyped, so missing values
                # arrive as NaN rather than None.
                stage = None
            outputs.append(
                Stage1Output(
                    subject_id=str(subject_id),
                    diagnosis=y[subject_id],
                    prediction_confidence=float(fit.proba[i]),
                    disease_stage=stage,
                    top_biomarkers=top_biomarkers,
                    mofa_factors=factors,
                    environmental_risk_score=self._env_score(environmental_scores, subject_id),
                    provenance=base_provenance,
                )
            )
        return outputs

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _env_score(environmental_scores: pd.Series, subject_id) -> float:
        """A missing exposure score must fail loudly.

        Defaulting to 0.0 would render as a measured "0.0/10" — the lowest
        exposure on the scale — for a subject whose exposure was never assessed.
        """
        value = environmental_scores.get(subject_id)
        if value is None or pd.isna(value):
            raise ValueError(
                f"no environmental risk score for subject {subject_id!r}; refusing "
                f"to fabricate 0.0 for an unmeasured exposure"
            )
        return float(value)

    @staticmethod
    def _z_scores(X: pd.DataFrame) -> pd.DataFrame:
        """Standardise each feature across the cohort; constant features become NaN."""
        std = X.std(ddof=0)
        return (X - X.mean()).div(std.where(std > 0))

    @staticmethod
    def _z_for(z_scores: pd.DataFrame, subject_id, feature: str) -> float | None:
        if feature not in z_scores.columns or subject_id not in z_scores.index:
            return None
        value = z_scores.at[subject_id, feature]
        return None if pd.isna(value) else float(value)

    def _provenance(self, X: pd.DataFrame, fit: EnsembleFit,
                    provenance: Provenance | None) -> Provenance:
        """Fill the pipeline-derived provenance fields, keeping caller-supplied ones."""
        supplied = provenance or Provenance()
        return Provenance(
            cohort_size=supplied.cohort_size or len(X),
            shap_out_of_fold=fit.out_of_fold,
            synthetic_modalities=supplied.synthetic_modalities,
            datasets=supplied.datasets,
            model=supplied.model or "XGBoost classifier with TreeSHAP attributions",
            cv_auc=fit.cv_auc if supplied.cv_auc is None else supplied.cv_auc,
        )

    @staticmethod
    def infer_modality(feature_name: str) -> str:
        """Best-effort modality assignment from a feature's name.

        Callers that know the true modality should prefix feature names on the
        way in (``microbiome:Prevotella``) rather than relying on this.
        """
        name = feature_name.lower()
        if RSID_PATTERN.search(name):
            return "genomics"
        prefix, sep, rest = name.partition(":")
        if sep:
            # The pipeline namespaces every column; trust the prefix and only
            # scan the remainder, so 'integrated:factor_1' cannot fall through
            # to 'clinical' just because the prefix hid the 'factor' stem.
            if prefix == "integrated":
                return "integrated"
            for modality, _patterns in MODALITY_PATTERNS:
                if prefix == modality:
                    return modality
            name = rest
        for modality, patterns in MODALITY_PATTERNS:
            if any(pattern in name for pattern in patterns):
                return modality
        if name.startswith("factor") or name.startswith("mofa"):
            return "integrated"
        return "clinical"
