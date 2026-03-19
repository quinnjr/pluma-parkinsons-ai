from __future__ import annotations
from dataclasses import dataclass, field
import json
from typing import Optional

VALID_DIAGNOSES = {"PD", "HC", "SWEDD", "Prodromal"}
VALID_STAGES = {"early", "mid", "late", None}


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
class BiomarkerHit:
    modality: str
    feature: str
    shap_value: float
    direction: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"modality": self.modality, "feature": self.feature, "shap": self.shap_value}
        if self.direction is not None:
            d["direction"] = self.direction
        return d

    @classmethod
    def from_dict(cls, d: dict) -> BiomarkerHit:
        return cls(
            modality=d["modality"],
            feature=d["feature"],
            shap_value=d["shap"],
            direction=d.get("direction"),
        )


@dataclass
class Stage1Output:
    subject_id: str
    diagnosis: str
    prediction_confidence: float
    disease_stage: Optional[str]
    top_biomarkers: list[BiomarkerHit]
    mofa_factors: dict[str, float]
    environmental_risk_score: float

    def __post_init__(self):
        if self.diagnosis not in VALID_DIAGNOSES:
            raise ValueError(f"Invalid diagnosis: {self.diagnosis!r}")
        if self.disease_stage not in VALID_STAGES:
            raise ValueError(f"Invalid disease_stage: {self.disease_stage!r}")

    def to_dict(self) -> dict:
        return {
            "subject_id": self.subject_id,
            "diagnosis": self.diagnosis,
            "prediction_confidence": self.prediction_confidence,
            "disease_stage": self.disease_stage,
            "top_biomarkers": [b.to_dict() for b in self.top_biomarkers],
            "mofa_factors": self.mofa_factors,
            "environmental_risk_score": self.environmental_risk_score,
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
        )
