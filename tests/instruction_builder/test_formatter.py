# tests/instruction_builder/test_formatter.py
import pytest
from src.models import Stage1Output, BiomarkerHit
from src.instruction_builder.formatter import InstructionFormatter

@pytest.fixture
def sample_output():
    return Stage1Output(
        subject_id="PD_001",
        diagnosis="PD",
        prediction_confidence=0.91,
        disease_stage="early",
        top_biomarkers=[
            BiomarkerHit("genomics", "LRRK2_p.G2019S", 0.34, "up"),
            BiomarkerHit("transcriptomics", "SNCA", 0.21, "up"),
            BiomarkerHit("microbiome", "Prevotella", -0.18, "down"),
        ],
        mofa_factors={"factor_1": 0.52, "factor_2": -0.31},
        environmental_risk_score=6.8,
    )

def test_biomarker_discovery_pair_has_required_keys(sample_output):
    formatter = InstructionFormatter()
    pair = formatter.biomarker_discovery(sample_output)
    assert set(pair.keys()) >= {"task", "instruction", "input", "output"}
    assert pair["task"] == "biomarker_discovery"

def test_biomarker_discovery_output_mentions_top_biomarker(sample_output):
    formatter = InstructionFormatter()
    pair = formatter.biomarker_discovery(sample_output)
    assert "LRRK2" in pair["output"]

def test_clinical_prediction_output_mentions_diagnosis(sample_output):
    formatter = InstructionFormatter()
    pair = formatter.clinical_prediction(sample_output)
    assert "PD" in pair["output"]
    assert "early" in pair["output"]

def test_clinical_prediction_output_mentions_confidence(sample_output):
    formatter = InstructionFormatter()
    pair = formatter.clinical_prediction(sample_output)
    # 0.91 * 100 = 91%
    assert "91" in pair["output"]

def test_cross_modal_synthesis_has_output(sample_output):
    formatter = InstructionFormatter()
    pair = formatter.cross_modal_synthesis(sample_output)
    assert len(pair["output"]) > 50
    assert pair["task"] == "cross_modal_synthesis"

def test_all_formats_returns_three_pairs(sample_output):
    formatter = InstructionFormatter()
    pairs = formatter.all_formats(sample_output)
    assert len(pairs) == 3
    tasks = {p["task"] for p in pairs}
    assert tasks == {"biomarker_discovery", "clinical_prediction", "cross_modal_synthesis"}

def test_profile_input_contains_biomarker_info(sample_output):
    formatter = InstructionFormatter()
    pair = formatter.biomarker_discovery(sample_output)
    assert "LRRK2_p.G2019S" in pair["input"]
    assert "environmental" in pair["input"].lower() or "6.8" in pair["input"]
