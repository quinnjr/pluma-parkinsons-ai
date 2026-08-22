"""Prompt templates for the three Stage-2 instruction tasks.

The system prompt is part of the training signal, not decoration: it is what
makes "I cannot support that from the curated evidence" an in-distribution
response rather than a failure mode.
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a Parkinson's disease multi-omics research assistant. You interpret "
    "the output of a machine-learning pipeline (MOFA+ latent factors, an XGBoost "
    "classifier, and TreeSHAP attributions) for a single subject.\n\n"
    "Rules you always follow:\n"
    "1. Separate what the model measured from what the literature reports. A SHAP "
    "value describes the classifier's behaviour, not biology.\n"
    "2. Cite a PMID for every claim about Parkinson's disease. If a feature has no "
    "curated annotation, say so and describe it as a statistical association only.\n"
    "3. Never invent a citation, a gene function, or an effect direction.\n"
    "4. State the limits of the evidence: cohort size, whether attributions were "
    "computed out of fold, and which modalities were simulated rather than measured.\n"
    "5. You are not making a clinical diagnosis. Frame predictions as model output."
)

BIOMARKER_INSTRUCTIONS = [
    "Given the following multi-omics profile, identify the most significant biomarkers "
    "and explain what is known about each. Distinguish curated literature findings from "
    "model-derived statistical associations.",
    "Analyse this integrated multi-omics profile and rank the features driving the model's "
    "Parkinson's disease signal. For each, state its biological function, the reported "
    "association in PD with a citation, and whether this subject's value is concordant "
    "with that report.",
    "Review this subject's multi-omics data and report which molecular features most "
    "strongly influenced the classifier. Flag any feature for which no curated PD "
    "annotation exists.",
]

PREDICTION_INSTRUCTIONS = [
    "Based on the integrated multi-omics and environmental profile below, report the "
    "model's Parkinson's disease prediction and disease stage, the evidence behind it, "
    "and how much confidence that evidence supports.",
    "Using the following multi-omics data, summarise the classifier's assessment of this "
    "subject and the key evidence for it. Be explicit about the limitations of the "
    "underlying cohort and attributions.",
    "Interpret this multi-omics profile: what does the model predict, on what evidence, "
    "and what would be needed to raise confidence in that prediction?",
]

SYNTHESIS_INSTRUCTIONS = [
    "How do the findings across different omics modalities relate to one another in this "
    "subject's profile? Synthesise the cross-modal evidence and note where the modalities "
    "disagree.",
    "Describe the biological picture suggested by the combination of modalities in this "
    "subject's profile, citing the literature where it exists and marking where it does not.",
    "What does the interplay between the genetic, transcriptomic, microbiome, and "
    "environmental features suggest about this subject? Identify convergent and "
    "conflicting signals.",
]

TASK_INSTRUCTIONS = {
    "biomarker_discovery": BIOMARKER_INSTRUCTIONS,
    "clinical_prediction": PREDICTION_INSTRUCTIONS,
    "cross_modal_synthesis": SYNTHESIS_INSTRUCTIONS,
}
