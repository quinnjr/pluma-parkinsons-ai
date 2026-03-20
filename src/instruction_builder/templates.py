# src/instruction_builder/templates.py

BIOMARKER_INSTRUCTIONS = [
    "Given the following multi-omics profile for a patient, identify the most significant biomarkers associated with Parkinson's disease and explain their biological relevance.",
    "Analyze this integrated multi-omics profile and rank the key biomarkers driving the Parkinson's disease signal. Explain the mechanistic role of each in PD pathophysiology.",
    "Review this patient's multi-omics data and identify which molecular features most strongly indicate Parkinson's disease risk or progression.",
]

PREDICTION_INSTRUCTIONS = [
    "Based on the integrated multi-omics and environmental profile below, predict this patient's Parkinson's disease diagnosis and disease stage. Provide your reasoning.",
    "Using the following multi-omics data, assess whether this patient has Parkinson's disease, and if so, at what stage. Explain the key evidence supporting your conclusion.",
    "Interpret this multi-omics profile to predict PD diagnosis and disease progression stage. Cite the most influential features in your reasoning.",
]

SYNTHESIS_INSTRUCTIONS = [
    "How do the findings across different omics modalities interact and converge in this patient's Parkinson's disease profile? Synthesize the cross-modal evidence.",
    "Describe the biological story told by the combination of genomic, transcriptomic, microbiome, and environmental findings in this patient profile.",
    "What does the interplay between genetic variants, gene expression, gut microbiome composition, and environmental exposures suggest about this patient's PD etiology?",
]
