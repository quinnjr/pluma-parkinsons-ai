import pytest

from src.knowledge import Citation, Entity, KnowledgeBase, PDAssociation
from src.training.model_utils import (
    audit_citations,
    biomarker_recall,
    compute_metrics,
    extract_diagnosis,
    extract_pmids,
    token_f1,
)


@pytest.fixture
def kb():
    citation = Citation(
        key="LRRK2_TEST",
        pmid="1111111",
        title="LRRK2 in PD",
        journal="Test J",
        year=2020,
        first_author="Doe J",
    )
    entity = Entity(
        key="LRRK2",
        aliases=("LRRK2", "G2019S"),
        modalities=("genomics",),
        function="Kinase.",
        pd_association=PDAssociation(
            measured_as="genotype",
            direction="up",
            statement="LRRK2 mutations cause PD.",
            citations=(citation,),
        ),
    )
    return KnowledgeBase(entities=(entity,), citations={citation.key: citation})


def test_extract_pmids_dedupes_in_order():
    text = "See PMID:123 and pmid: 456, also PMID:123 again."
    assert extract_pmids(text) == ["123", "456"]


def test_extract_pmids_empty_text():
    assert extract_pmids("") == []
    assert extract_pmids(None) == []


def test_audit_citations_flags_unknown_pmids(kb):
    audit = audit_citations("Supported [PMID:1111111], invented [PMID:9999999].", kb)
    assert audit.supported == ("1111111",)
    assert audit.hallucinated == ("9999999",)
    assert audit.hallucination_rate == 0.5


def test_audit_citations_no_citations_is_not_hallucination(kb):
    audit = audit_citations("No claims made here.", kb)
    assert audit.cited == ()
    assert audit.hallucination_rate == 0.0


def test_token_f1_identical_is_one():
    assert token_f1("LRRK2 pushes toward PD", "LRRK2 pushes toward PD") == 1.0


def test_token_f1_disjoint_is_zero():
    assert token_f1("alpha beta", "gamma delta") == 0.0


def test_biomarker_recall_counts_named_features():
    text = "The top features are LRRK2 and SNCA_expr."
    assert biomarker_recall(text, ["LRRK2", "SNCA_expr", "GBA1"]) == pytest.approx(2 / 3)


def test_biomarker_recall_empty_expected_is_zero():
    assert biomarker_recall("anything", []) == 0.0


def test_extract_diagnosis_parses_model_output_line():
    assert extract_diagnosis("Model output: PD, staged early, at 91.0%") == "PD"
    assert extract_diagnosis("model output: prodromal, at 40%") == "Prodromal"
    assert extract_diagnosis("no diagnosis line here") is None


def test_compute_metrics_empty():
    assert compute_metrics([], []) == {"n_samples": 0}


def test_compute_metrics_length_mismatch_raises(kb):
    with pytest.raises(ValueError):
        compute_metrics(["a"], ["a", "b"], knowledge_base=kb)


def test_compute_metrics_has_no_exact_match(kb):
    metrics = compute_metrics(["PD early"], ["PD early"], knowledge_base=kb)
    assert "exact_match" not in metrics
    assert metrics["token_f1"] == 1.0


def test_compute_metrics_aggregates(kb):
    records = [
        {
            "task": "clinical_prediction",
            "grounding": {"annotated_features": ["LRRK2"]},
        },
        {
            "task": "biomarker_discovery",
            "grounding": {"annotated_features": ["SNCA_expr"]},
        },
    ]
    predictions = [
        "Model output: PD. LRRK2 drives it [PMID:1111111].",
        "SNCA_expr ranked first [PMID:9999999].",
    ]
    references = [
        "Model output: PD. LRRK2 [PMID:1111111].",
        "SNCA_expr ranked first [PMID:1111111].",
    ]
    metrics = compute_metrics(predictions, references, records, knowledge_base=kb)
    assert metrics["n_samples"] == 2
    assert metrics["citation_hallucination_rate"] == pytest.approx(0.5)
    assert metrics["responses_with_citations"] == 1.0
    assert metrics["biomarker_recall"] == 1.0
    assert metrics["diagnosis_accuracy"] == 1.0
    assert metrics["diagnosis_parse_rate"] == 1.0


def test_compute_metrics_diagnosis_only_on_prediction_task(kb):
    records = [{"task": "biomarker_discovery"}]
    metrics = compute_metrics(["Model output: PD"], ["Model output: HC"], records,
                              knowledge_base=kb)
    assert "diagnosis_accuracy" not in metrics
