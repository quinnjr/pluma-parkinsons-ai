# src/instruction_builder/formatter.py
from __future__ import annotations
import random
from src.models import Stage1Output
from src.instruction_builder.templates import (
    BIOMARKER_INSTRUCTIONS,
    PREDICTION_INSTRUCTIONS,
    SYNTHESIS_INSTRUCTIONS,
)


class InstructionFormatter:
    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def _format_profile(self, output: Stage1Output) -> str:
        lines = [f"Subject: {output.subject_id}"]
        lines.append(f"Environmental risk score: {output.environmental_risk_score:.1f}/10")
        lines.append("Top biomarkers by SHAP importance:")
        for b in output.top_biomarkers:
            direction = f" ({b.direction})" if b.direction else ""
            lines.append(
                f"  - [{b.modality}] {b.feature}{direction}: SHAP={b.shap_value:.3f}"
            )
        if output.mofa_factors:
            top_factors = sorted(
                output.mofa_factors.items(), key=lambda x: abs(x[1]), reverse=True
            )[:3]
            lines.append("MOFA+ latent factors (top 3 by magnitude):")
            for fname, fval in top_factors:
                lines.append(f"  - {fname}: {fval:.3f}")
        return "\n".join(lines)

    def biomarker_discovery(self, output: Stage1Output) -> dict:
        instruction = self._rng.choice(BIOMARKER_INSTRUCTIONS)
        top = output.top_biomarkers[:5]
        response_lines = [
            f"The most significant biomarkers in this {output.diagnosis} profile are:",
        ]
        for i, b in enumerate(top, 1):
            response_lines.append(
                f"{i}. {b.feature} [{b.modality}] (SHAP={b.shap_value:.3f},"
                f" direction={b.direction}): "
                f"This feature is a known contributor to PD pathophysiology."
            )
        return {
            "task": "biomarker_discovery",
            "instruction": instruction,
            "input": self._format_profile(output),
            "output": "\n".join(response_lines),
        }

    def clinical_prediction(self, output: Stage1Output) -> dict:
        instruction = self._rng.choice(PREDICTION_INSTRUCTIONS)
        conf_pct = output.prediction_confidence * 100
        stage_str = f", {output.disease_stage} stage" if output.disease_stage else ""
        response = (
            f"Prediction: {output.diagnosis}{stage_str} "
            f"(confidence: {conf_pct:.1f}%).\n\n"
            f"Key supporting evidence:\n"
        )
        for b in output.top_biomarkers[:3]:
            response += f"- {b.feature} ({b.modality}): SHAP contribution {b.shap_value:.3f}\n"
        return {
            "task": "clinical_prediction",
            "instruction": instruction,
            "input": self._format_profile(output),
            "output": response,
        }

    def cross_modal_synthesis(self, output: Stage1Output) -> dict:
        instruction = self._rng.choice(SYNTHESIS_INSTRUCTIONS)
        modalities = list({b.modality for b in output.top_biomarkers})
        b0, b1 = output.top_biomarkers[0], output.top_biomarkers[1]
        response = (
            f"This profile shows convergent signals across {len(modalities)} modalities "
            f"({', '.join(modalities)}), collectively supporting a {output.diagnosis} classification.\n\n"
            f"The strongest cross-modal interaction appears between "
            f"{b0.feature} ({b0.modality}) and {b1.feature} ({b1.modality}), "
            f"both contributing in the same direction to the PD signal."
        )
        return {
            "task": "cross_modal_synthesis",
            "instruction": instruction,
            "input": self._format_profile(output),
            "output": response,
        }

    def all_formats(self, output: Stage1Output) -> list[dict]:
        return [
            self.biomarker_discovery(output),
            self.clinical_prediction(output),
            self.cross_modal_synthesis(output),
        ]
