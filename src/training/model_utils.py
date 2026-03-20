# src/training/model_utils.py
from __future__ import annotations


def format_prompt(pair: dict) -> str:
    """Format an instruction-response pair using Alpaca-style template.

    Compatible with Mistral's instruction-tuning format. The template
    separates instruction, optional input context, and response clearly,
    making it easy to run inference with the same template post-training.
    """
    instruction = pair["instruction"]
    input_text = pair.get("input", "").strip()
    response = pair.get("output", "")

    if input_text:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{response}"
        )
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{response}"
    )


def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    """Compute exact-match accuracy between predictions and references."""
    if not predictions:
        return {"exact_match": 0.0, "n_samples": 0}
    exact = sum(p.strip() == r.strip() for p, r in zip(predictions, references))
    return {
        "exact_match": exact / len(predictions),
        "n_samples": len(predictions),
    }
