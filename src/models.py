from __future__ import annotations

import json
from dataclasses import dataclass, field, replace

VALID_DIAGNOSES = {"PD", "HC", "SWEDD", "Prodromal"}
VALID_STAGES = {"early", "mid", "late", None}

#: A SHAP attribution says which way a feature pushed *the model*, not which way
#: the measurement moved. Keeping the two apart is the whole point.
TOWARD_PD = "toward_pd"
AWAY_FROM_PD = "away_from_pd"
VALID_EFFECTS = {TOWARD_PD, AWAY_FROM_PD}


@dataclass
class Subject:
    subject_id: str
    diagnosis: str
    age: int
    sex: str

    def __post_init__(self):
        if self.diagnosis not in VALID_DIAGNOSES:
            raise ValueError(f"diagnosis must be one of {VALID_DIAGNOSES}, got {self.diagnosis!r}")


@dataclass
class OmicsProfile:
    modality: str
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class Provenance:
    """How a Stage1Output was produced, so downstream text can be honest about it.

    Without this, generated training targets have no way to distinguish a
    biomarker ranked on 1,200 real subjects with out-of-fold SHAP from one
    ranked on 20 subjects with in-sample SHAP over simulated features.
    """

    cohort_size: int = 0
    #: False means SHAP was computed on the same rows the model was fitted on,
    #: which inflates apparent importance. Reported in every caveat block.
    shap_out_of_fold: bool = False
    #: Modalities whose values were simulated rather than measured.
    synthetic_modalities: tuple[str, ...] = ()
    #: Source accessions/cohorts, e.g. ("GSE6613", "NHANES 2017-2018").
    datasets: tuple[str, ...] = ()
    model: str = ""
    #: Cross-validated ROC AUC of the classifier, if it was estimated.
    cv_auc: float | None = None

    def to_dict(self) -> dict:
        return {
            "cohort_size": self.cohort_size,
            "shap_out_of_fold": self.shap_out_of_fold,
            "synthetic_modalities": list(self.synthetic_modalities),
            "datasets": list(self.datasets),
            "model": self.model,
            "cv_auc": self.cv_auc,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> Provenance:
        if not d:
            return cls()
        return cls(
            cohort_size=d.get("cohort_size", 0),
            shap_out_of_fold=d.get("shap_out_of_fold", False),
            synthetic_modalities=tuple(d.get("synthetic_modalities", ())),
            datasets=tuple(d.get("datasets", ())),
            model=d.get("model", ""),
            cv_auc=d.get("cv_auc"),
        )


@dataclass
class BiomarkerHit:
    modality: str
    feature: str
    shap_value: float
    #: Which way this feature pushed the classifier for this subject.
    effect: str = TOWARD_PD
    #: The subject's standardised value for this feature (z-score against the
    #: cohort). ``None`` when the cohort statistics were not available.
    value_z: float | None = None

    def __post_init__(self):
        if self.effect not in VALID_EFFECTS:
            raise ValueError(f"effect must be one of {sorted(VALID_EFFECTS)}, got {self.effect!r}")

    @property
    def pushes_toward_pd(self) -> bool:
        return self.effect == TOWARD_PD

    @property
    def observed_direction(self) -> str | None:
        """``"up"``/``"down"`` for the measurement itself, or ``None`` if unknown.

        A z of exactly 0 has no direction; calling it "down" would assert an
        observation that was not made.
        """
        if self.value_z is None or self.value_z == 0:
            return None
        return "up" if self.value_z > 0 else "down"

    def to_dict(self) -> dict:
        d = {
            "modality": self.modality,
            "feature": self.feature,
            "shap": self.shap_value,
            "effect": self.effect,
        }
        if self.value_z is not None:
            d["value_z"] = self.value_z
        return d

    @classmethod
    def from_dict(cls, d: dict) -> BiomarkerHit:
        # Older Stage1Output JSON stored the SHAP sign under "direction" as
        # up/down. Read it, but map it onto the unambiguous field name.
        effect = d.get("effect")
        if effect is None:
            legacy = d.get("direction")
            effect = AWAY_FROM_PD if legacy == "down" else TOWARD_PD
        return cls(
            modality=d["modality"],
            feature=d["feature"],
            shap_value=d["shap"],
            effect=effect,
            value_z=d.get("value_z"),
        )


@dataclass
class Stage1Output:
    subject_id: str
    diagnosis: str
    prediction_confidence: float
    disease_stage: str | None
    top_biomarkers: list[BiomarkerHit]
    mofa_factors: dict[str, float]
    environmental_risk_score: float
    provenance: Provenance = field(default_factory=Provenance)

    def __post_init__(self):
        if self.diagnosis not in VALID_DIAGNOSES:
            raise ValueError(f"Invalid diagnosis: {self.diagnosis!r}")
        if self.disease_stage not in VALID_STAGES:
            raise ValueError(f"Invalid disease_stage: {self.disease_stage!r}")
        if not 0.0 <= self.prediction_confidence <= 1.0:
            raise ValueError(
                f"prediction_confidence must be a probability in [0, 1], "
                f"got {self.prediction_confidence!r}"
            )

    @property
    def modalities(self) -> list[str]:
        """Modalities represented in the top biomarkers, in rank order."""
        seen: list[str] = []
        for hit in self.top_biomarkers:
            if hit.modality not in seen:
                seen.append(hit.modality)
        return seen

    def with_provenance(self, provenance: Provenance) -> Stage1Output:
        return replace(self, provenance=provenance)

    def to_dict(self) -> dict:
        return {
            "subject_id": self.subject_id,
            "diagnosis": self.diagnosis,
            "prediction_confidence": self.prediction_confidence,
            "disease_stage": self.disease_stage,
            "top_biomarkers": [b.to_dict() for b in self.top_biomarkers],
            "mofa_factors": self.mofa_factors,
            "environmental_risk_score": self.environmental_risk_score,
            "provenance": self.provenance.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> Stage1Output:
        return cls(
            subject_id=d["subject_id"],
            diagnosis=d["diagnosis"],
            prediction_confidence=d["prediction_confidence"],
            disease_stage=d.get("disease_stage"),
            top_biomarkers=[BiomarkerHit.from_dict(b) for b in d.get("top_biomarkers", [])],
            mofa_factors=d.get("mofa_factors", {}),
            environmental_risk_score=d["environmental_risk_score"],
            provenance=Provenance.from_dict(d.get("provenance")),
        )
