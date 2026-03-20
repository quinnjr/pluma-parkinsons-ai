# tests/training/test_model_utils.py
import pytest
from src.training.model_utils import format_prompt, compute_metrics

def test_format_prompt_with_input():
    pair = {
        "instruction": "Identify biomarkers in this profile.",
        "input": "Subject: PD_001\nLRRK2: SHAP=0.34",
        "output": "The top biomarker is LRRK2.",
    }
    prompt = format_prompt(pair)
    assert "### Instruction:" in prompt
    assert "### Input:" in prompt
    assert "### Response:" in prompt
    assert "LRRK2" in prompt
    assert "Identify biomarkers" in prompt

def test_format_prompt_without_input():
    pair = {
        "instruction": "What is Parkinson's disease?",
        "input": "",
        "output": "Parkinson's disease is...",
    }
    prompt = format_prompt(pair)
    assert "### Input:" not in prompt
    assert "### Instruction:" in prompt
    assert "### Response:" in prompt

def test_format_prompt_missing_input_key():
    pair = {
        "instruction": "Explain this.",
        "output": "Sure.",
    }
    prompt = format_prompt(pair)
    assert "### Input:" not in prompt
    assert "### Instruction:" in prompt

def test_compute_metrics_exact_match():
    metrics = compute_metrics(
        predictions=["PD early", "HC", "PD mid"],
        references=["PD early", "HC", "PD late"],
    )
    assert isinstance(metrics, dict)
    assert "exact_match" in metrics
    assert abs(metrics["exact_match"] - 2/3) < 1e-9

def test_compute_metrics_empty():
    metrics = compute_metrics(predictions=[], references=[])
    assert metrics["exact_match"] == 0.0
    assert metrics["n_samples"] == 0
