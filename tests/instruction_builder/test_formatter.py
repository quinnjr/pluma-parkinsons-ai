import pytest

from src.instruction_builder.formatter import (
    CONCORDANT,
    DISCORDANT,
    InstructionFormatter,
)
from src.knowledge import Citation, Entity, KnowledgeBase, PDAssociation
from src.models import BiomarkerHit, Provenance, Stage1Output


@pytest.fixture
def kb():
    lrrk2_cit = Citation(key="LRRK2_TEST", pmid="1111111", title="LRRK2 in PD",
                         journal="Test J", year=2020, first_author="Doe J")
    prev_cit = Citation(key="PREV_TEST", pmid="2222222", title="Gut microbiome in PD",
                        journal="Test J", year=2021, first_author="Roe R")
    entities = (
        Entity(
            key="LRRK2",
            aliases=("LRRK2", "G2019S", "rs34637584"),
            modalities=("genomics", "transcriptomics"),
            function="Multidomain kinase.",
            pd_association=PDAssociation(
                measured_as="genotype",
                direction="up",
                statement="LRRK2 mutations cause autosomal-dominant parkinsonism.",
                citations=(lrrk2_cit,),
            ),
        ),
        Entity(
            key="PREVOTELLA",
            aliases=("Prevotella", "g__Prevotella"),
            modalities=("microbiome",),
            function="Gut commensal genus.",
            pd_association=PDAssociation(
                measured_as="relative abundance",
                direction="down",
                statement="Prevotellaceae abundance is reduced in PD faecal microbiota.",
                citations=(prev_cit,),
            ),
        ),
        Entity(
            key="ACTB",
            aliases=("ACTB",),
            modalities=("transcriptomics",),
            function="Cytoskeletal actin.",
        ),
    )
    return KnowledgeBase(entities=entities,
                         citations={c.key: c for c in (lrrk2_cit, prev_cit)})


@pytest.fixture
def formatter(kb):
    return InstructionFormatter(seed=0, knowledge_base=kb)


@pytest.fixture
def sample_output():
    return Stage1Output(
        subject_id="PD_001",
        diagnosis="PD",
        prediction_confidence=0.91,
        disease_stage="early",
        top_biomarkers=[
            BiomarkerHit("genomics", "LRRK2_G2019S", 0.34, "toward_pd", value_z=1.4),
            BiomarkerHit("microbiome", "g__Prevotella", -0.18, "away_from_pd", value_z=0.8),
            BiomarkerHit("transcriptomics", "RANDOM_PROBE_1552281_at", 0.21, "toward_pd",
                         value_z=-0.3),
        ],
        mofa_factors={"factor_1": 0.52, "factor_2": -0.31},
        environmental_risk_score=6.8,
        provenance=Provenance(
            cohort_size=50,
            shap_out_of_fold=False,
            synthetic_modalities=("microbiome",),
            datasets=("GSE6613",),
            model="XGBoost",
            cv_auc=0.65,
        ),
    )


def test_pair_has_required_keys(formatter, sample_output):
    pair = formatter.biomarker_discovery(sample_output)
    assert set(pair.keys()) >= {"task", "subject_id", "instruction", "input", "output",
                                "grounding"}
    assert pair["task"] == "biomarker_discovery"
    assert pair["subject_id"] == "PD_001"


def test_grounding_separates_annotated_from_unannotated(formatter, sample_output):
    pair = formatter.biomarker_discovery(sample_output)
    grounding = pair["grounding"]
    assert set(grounding["annotated_features"]) == {"LRRK2_G2019S", "g__Prevotella"}
    assert grounding["unannotated_features"] == ["RANDOM_PROBE_1552281_at"]
    assert grounding["pmids"] == ["1111111", "2222222"]
    assert grounding["synthetic_modalities"] == ["microbiome"]


def test_annotated_feature_is_cited(formatter, sample_output):
    output = formatter.biomarker_discovery(sample_output)["output"]
    assert "LRRK2 mutations cause autosomal-dominant parkinsonism." in output
    assert "PMID:1111111" in output
    assert "References:" in output


def test_unannotated_feature_gets_abstention_not_fabrication(formatter, sample_output):
    output = formatter.biomarker_discovery(sample_output)["output"]
    assert "No curated annotation" in output
    # The failure mode the rewrite removed: a canned mechanism claim attached
    # to every feature regardless of evidence.
    assert "known contributor" not in output.lower()


def test_all_cited_pmids_come_from_the_kb(formatter, sample_output):
    from src.training.model_utils import audit_citations

    for pair in formatter.all_formats(sample_output):
        audit = audit_citations(pair["output"], formatter.kb)
        assert audit.hallucinated == ()


def test_caveats_report_in_sample_shap_and_synthetic_modalities(formatter, sample_output):
    output = formatter.biomarker_discovery(sample_output)["output"]
    assert "Limitations:" in output
    assert "in-sample" in output
    assert "simulated" in output
    assert "50 subjects" in output


def test_profile_input_reports_provenance(formatter, sample_output):
    text = formatter.biomarker_discovery(sample_output)["input"]
    assert "LRRK2_G2019S" in text
    assert "GSE6613" in text
    assert "in-sample" in text
    assert "6.8" in text


def test_concordance_verdicts(formatter, sample_output):
    grounded = formatter.ground(sample_output)
    by_feature = {g.feature: g for g in grounded}
    # LRRK2: literature says up, observed z=+1.4 -> concordant.
    assert by_feature["LRRK2_G2019S"].concordance == CONCORDANT
    # Prevotella: literature says down, observed z=+0.8 -> discordant.
    assert by_feature["g__Prevotella"].concordance == DISCORDANT


def test_discordant_hit_is_flagged_in_text(formatter, sample_output):
    output = formatter.biomarker_discovery(sample_output)["output"]
    assert "opposite" in output


def test_clinical_prediction_reports_the_models_call(formatter, sample_output):
    output = formatter.clinical_prediction(sample_output)["output"]
    assert "Model output: PD" in output
    assert "91.0%" in output
    assert "not a clinical diagnosis" in output


def test_clinical_prediction_never_reports_the_label(formatter, sample_output):
    # A PD-labelled subject the classifier scored at 0.08 must be reported as
    # the classifier's HC call, not the recorded diagnosis.
    import dataclasses
    misclassified = dataclasses.replace(sample_output, prediction_confidence=0.08)
    output = formatter.clinical_prediction(misclassified)["output"]
    assert "Model output: HC" in output
    assert "8.0%" in output


def test_recorded_diagnosis_is_not_in_the_prompt(formatter, sample_output):
    for pair in formatter.all_formats(sample_output):
        assert "Recorded diagnosis" not in pair["input"]
        assert "diagnosis: PD" not in pair["input"]


def test_cross_modal_synthesis_excludes_unannotated(formatter, sample_output):
    output = formatter.cross_modal_synthesis(sample_output)["output"]
    assert "Excluded from the synthesis" in output
    assert "RANDOM_PROBE_1552281_at" in output


def test_cross_modal_synthesis_abstains_with_no_annotations(formatter):
    bare = Stage1Output(
        subject_id="HC_009",
        diagnosis="HC",
        prediction_confidence=0.12,
        disease_stage=None,
        top_biomarkers=[
            BiomarkerHit("transcriptomics", "PROBE_A_at", 0.1, "toward_pd"),
            BiomarkerHit("transcriptomics", "PROBE_B_at", -0.05, "away_from_pd"),
        ],
        mofa_factors={},
        environmental_risk_score=1.0,
    )
    pair = formatter.cross_modal_synthesis(bare)
    assert "no cross-modal biological synthesis can be offered" in pair["output"]
    assert "References: none." in pair["output"]
    assert pair["grounding"]["pmids"] == []


def test_all_formats_returns_three_tasks(formatter, sample_output):
    pairs = formatter.all_formats(sample_output)
    assert {p["task"] for p in pairs} == {
        "biomarker_discovery", "clinical_prediction", "cross_modal_synthesis"
    }


def test_references_cover_the_full_profile_not_the_task_cutoff(kb):
    # 5 unannotated features rank above the one annotated (LRRK2, 6th). The
    # clinical task discusses only the top 3, but the prompt lists all 6, so
    # claiming "References: none" would be false of what the model sees.
    hits = [
        BiomarkerHit("transcriptomics", f"PROBE_{i}_at", 0.9 - i * 0.1, "toward_pd")
        for i in range(5)
    ] + [BiomarkerHit("genomics", "LRRK2", 0.05, "toward_pd", value_z=1.0)]
    output = Stage1Output(
        subject_id="PD_002",
        diagnosis="PD",
        prediction_confidence=0.8,
        disease_stage=None,
        top_biomarkers=hits,
        mofa_factors={},
        environmental_risk_score=3.0,
    )
    formatter = InstructionFormatter(seed=0, knowledge_base=kb)
    pair = formatter.clinical_prediction(output)
    assert "References: none" not in pair["output"]
    assert "PMID:1111111" in pair["output"]
    assert "5 of the 6 features listed have no curated" in pair["output"]
    # annotated_features covers only the NARRATED hits (top 3 here, all
    # unannotated) — biomarker_recall scores against what the body names, so
    # including LRRK2 (ranked 6th, never narrated) would make the reference
    # response itself unable to reach recall 1.0. The full-profile grounding
    # still drives the references and caveats asserted above.
    assert pair["grounding"]["annotated_features"] == []
    assert pair["grounding"]["pmids"] == ["1111111"]


def test_synthetic_caveat_fires_even_when_no_synthetic_feature_ranks(kb):
    output = Stage1Output(
        subject_id="PD_003",
        diagnosis="PD",
        prediction_confidence=0.9,
        disease_stage=None,
        top_biomarkers=[BiomarkerHit("genomics", "LRRK2", 0.3, "toward_pd")],
        mofa_factors={},
        environmental_risk_score=2.0,
        provenance=Provenance(cohort_size=200, shap_out_of_fold=True,
                              synthetic_modalities=("metabolomics",), cv_auc=0.97),
    )
    formatter = InstructionFormatter(seed=0, knowledge_base=kb)
    pair = formatter.biomarker_discovery(output)
    assert "simulated, not measured" in pair["output"]
    assert "inflated" in pair["output"]
    assert "inflated" in pair["input"]  # the AUC line carries the qualifier too


def test_zero_z_gets_no_direction(kb):
    output = Stage1Output(
        subject_id="PD_004",
        diagnosis="PD",
        prediction_confidence=0.9,
        disease_stage=None,
        top_biomarkers=[BiomarkerHit("genomics", "LRRK2", 0.3, "toward_pd", value_z=0.0)],
        mofa_factors={},
        environmental_risk_score=2.0,
    )
    formatter = InstructionFormatter(seed=0, knowledge_base=kb)
    grounded = formatter.ground(output)
    assert grounded[0].concordance == "undetermined"
    text = formatter.biomarker_discovery(output)["output"]
    assert "cohort mean" in text
    assert "reduced" not in text


def test_instruction_choice_is_seeded(kb, sample_output):
    a = InstructionFormatter(seed=7, knowledge_base=kb).all_formats(sample_output)
    b = InstructionFormatter(seed=7, knowledge_base=kb).all_formats(sample_output)
    assert [p["instruction"] for p in a] == [p["instruction"] for p in b]


def test_cross_modal_never_claims_convergence_within_one_modality(kb):
    # Both annotated hits are microbiome: no cross-modal claim may be made.
    output = Stage1Output(
        subject_id="PD_005",
        diagnosis="PD",
        prediction_confidence=0.9,
        disease_stage=None,
        top_biomarkers=[
            BiomarkerHit("microbiome", "g__Prevotella", -0.4, "away_from_pd", value_z=-0.5),
            BiomarkerHit("microbiome", "bug_Prevotella_2", -0.3, "away_from_pd"),
        ],
        mofa_factors={},
        environmental_risk_score=2.0,
    )
    formatter = InstructionFormatter(seed=0, knowledge_base=kb)
    text = formatter.cross_modal_synthesis(output)["output"]
    assert "Converging attributions across modalities" not in text
    assert "single modality" in text


def test_cross_modal_pairs_first_hit_with_distinct_modality(kb):
    # Highest annotated hit is microbiome; the partner must skip the second
    # microbiome hit and pair with the genomics one.
    output = Stage1Output(
        subject_id="PD_006",
        diagnosis="PD",
        prediction_confidence=0.9,
        disease_stage=None,
        top_biomarkers=[
            BiomarkerHit("microbiome", "g__Prevotella", -0.5, "away_from_pd"),
            BiomarkerHit("microbiome", "g__Prevotella_b", -0.4, "away_from_pd"),
            BiomarkerHit("genomics", "LRRK2", 0.3, "toward_pd", value_z=1.0),
        ],
        mofa_factors={},
        environmental_risk_score=2.0,
    )
    formatter = InstructionFormatter(seed=0, knowledge_base=kb)
    text = formatter.cross_modal_synthesis(output)["output"]
    assert "g__Prevotella [microbiome] and LRRK2 [genomics]" in text
    assert "opposite directions" in text


def test_catch_all_modality_does_not_strip_annotation(kb):
    # A curated feature whose modality could only be inferred as 'clinical'
    # (or 'integrated') must still resolve against the KB.
    output = Stage1Output(
        subject_id="PD_007",
        diagnosis="PD",
        prediction_confidence=0.9,
        disease_stage=None,
        top_biomarkers=[BiomarkerHit("clinical", "LRRK2_G2019S", 0.3, "toward_pd")],
        mofa_factors={},
        environmental_risk_score=2.0,
    )
    formatter = InstructionFormatter(seed=0, knowledge_base=kb)
    grounded = formatter.ground(output)
    assert grounded[0].is_annotated
