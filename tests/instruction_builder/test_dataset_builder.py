import pytest
from pathlib import Path
from src.models import Stage1Output, BiomarkerHit
from src.instruction_builder.dataset_builder import DatasetBuilder
from src.utils import load_jsonl

def _make_output(subject_id: str, diagnosis: str) -> Stage1Output:
    return Stage1Output(
        subject_id=subject_id,
        diagnosis=diagnosis,
        prediction_confidence=0.9 if diagnosis == "PD" else 0.1,
        disease_stage="early" if diagnosis == "PD" else None,
        top_biomarkers=[
            BiomarkerHit("genomics", "LRRK2", 0.3, "up"),
            BiomarkerHit("transcriptomics", "SNCA", 0.2, "up"),
        ],
        mofa_factors={"factor_1": 0.5},
        environmental_risk_score=5.0,
    )

@pytest.fixture
def outputs():
    return (
        [_make_output(f"PD_{i}", "PD") for i in range(30)]
        + [_make_output(f"HC_{i}", "HC") for i in range(30)]
    )

def test_build_pairs_triples_count(outputs):
    builder = DatasetBuilder()
    pairs = builder.build_pairs(outputs)
    assert len(pairs) == 60 * 3

def test_split_proportions(outputs):
    builder = DatasetBuilder(seed=42)
    pairs = builder.build_pairs(outputs)
    splits = builder.split(pairs)
    total = sum(len(v) for v in splits.values())
    assert total == len(pairs)
    assert len(splits["train"]) > len(splits["val"])
    assert len(splits["val"]) >= len(splits["test"])

def test_split_keys(outputs):
    builder = DatasetBuilder()
    pairs = builder.build_pairs(outputs)
    splits = builder.split(pairs)
    assert set(splits.keys()) == {"train", "val", "test"}

def test_save_creates_jsonl_files(outputs, tmp_path):
    builder = DatasetBuilder(seed=42)
    pairs = builder.build_pairs(outputs)
    splits = builder.split(pairs)
    builder.save(splits, tmp_path)
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "val.jsonl").exists()
    assert (tmp_path / "test.jsonl").exists()

def test_saved_jsonl_is_loadable(outputs, tmp_path):
    builder = DatasetBuilder(seed=42)
    pairs = builder.build_pairs(outputs)
    splits = builder.split(pairs)
    builder.save(splits, tmp_path)
    train = load_jsonl(tmp_path / "train.jsonl")
    assert len(train) == len(splits["train"])
    assert "instruction" in train[0]
    assert "output" in train[0]
