import pytest

from src.instruction_builder.dataset_builder import DatasetBuilder
from src.models import BiomarkerHit, Stage1Output
from src.utils import load_jsonl


def _make_output(subject_id: str, diagnosis: str) -> Stage1Output:
    return Stage1Output(
        subject_id=subject_id,
        diagnosis=diagnosis,
        prediction_confidence=0.9 if diagnosis == "PD" else 0.1,
        disease_stage="early" if diagnosis == "PD" else None,
        top_biomarkers=[
            BiomarkerHit("genomics", "LRRK2", 0.3, "toward_pd", value_z=1.0),
            BiomarkerHit("transcriptomics", "SNCA_expr", 0.2, "toward_pd", value_z=0.5),
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
    pairs = DatasetBuilder().build_pairs(outputs)
    assert len(pairs) == 60 * 3


def test_invalid_fractions_rejected():
    with pytest.raises(ValueError):
        DatasetBuilder(train_frac=0.9, val_frac=0.2)
    with pytest.raises(ValueError):
        DatasetBuilder(train_frac=0.0)


def test_split_proportions(outputs):
    builder = DatasetBuilder(seed=42)
    pairs = builder.build_pairs(outputs)
    splits = builder.split(pairs)
    assert sum(len(v) for v in splits.values()) == len(pairs)
    assert set(splits.keys()) == {"train", "val", "test"}
    assert len(splits["train"]) > len(splits["val"]) > 0
    assert len(splits["test"]) > 0


def test_split_is_by_subject_not_by_pair(outputs):
    builder = DatasetBuilder(seed=42)
    splits = builder.split(builder.build_pairs(outputs))
    subject_sets = {name: {p["subject_id"] for p in pairs}
                    for name, pairs in splits.items()}
    # A subject's three near-identical task pairs must never straddle a split
    # boundary, or validation loss measures memorisation.
    assert subject_sets["train"] & subject_sets["val"] == set()
    assert subject_sets["train"] & subject_sets["test"] == set()
    assert subject_sets["val"] & subject_sets["test"] == set()
    for name, pairs in splits.items():
        assert len(pairs) == 3 * len(subject_sets[name])


def test_split_is_deterministic_for_a_seed(outputs):
    pairs = DatasetBuilder(seed=7).build_pairs(outputs)
    a = DatasetBuilder(seed=7).split(pairs)
    b = DatasetBuilder(seed=7).split(pairs)
    assert {k: [p["subject_id"] for p in v] for k, v in a.items()} == \
           {k: [p["subject_id"] for p in v] for k, v in b.items()}


def test_tiny_cohort_still_gets_all_three_splits():
    outputs = [_make_output(f"S_{i}", "PD" if i % 2 else "HC") for i in range(3)]
    builder = DatasetBuilder(seed=1)
    splits = builder.split(builder.build_pairs(outputs))
    assert all(len(v) > 0 for v in splits.values())


def test_save_round_trips_jsonl(outputs, tmp_path):
    builder = DatasetBuilder(seed=42)
    splits = builder.split(builder.build_pairs(outputs))
    builder.save(splits, tmp_path)
    for name in ("train", "val", "test"):
        records = load_jsonl(tmp_path / f"{name}.jsonl")
        assert len(records) == len(splits[name])
    train = load_jsonl(tmp_path / "train.jsonl")
    assert {"instruction", "input", "output", "grounding"} <= set(train[0])


def test_fewer_than_three_subjects_raises():
    outputs = [_make_output("S_0", "PD"), _make_output("S_1", "HC")]
    builder = DatasetBuilder(seed=1)
    with pytest.raises(ValueError, match="at least 3 subjects"):
        builder.split(builder.build_pairs(outputs))


def test_extreme_fractions_still_yield_nonempty_test():
    # train 0.1 / val 0.8 passes the constructor; the split must still reserve
    # at least one subject for test at small n.
    for n in (3, 4, 5):
        outputs = [_make_output(f"S_{i}", "PD" if i % 2 else "HC") for i in range(n)]
        builder = DatasetBuilder(train_frac=0.1, val_frac=0.8, seed=1)
        splits = builder.split(builder.build_pairs(outputs))
        assert all(len(v) > 0 for v in splits.values()), (n, {
            k: len(v) for k, v in splits.items()})
