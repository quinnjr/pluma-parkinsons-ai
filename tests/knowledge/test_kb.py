import pytest

from src.knowledge import load_knowledge_base
from src.knowledge.kb import _load_citations, _load_entities


@pytest.fixture(scope="module")
def kb():
    return load_knowledge_base()


# -- bundled data integrity -----------------------------------------------------

def test_bundle_loads_and_validates(kb):
    assert len(kb.entities) >= 20
    assert len(kb.citations) >= 30


def test_every_citation_has_numeric_pmid(kb):
    for citation in kb.citations.values():
        assert citation.pmid.isdigit()
        assert citation.title and citation.first_author and citation.journal
        assert 1950 <= citation.year <= 2026


def test_every_pd_claim_is_cited(kb):
    for entity in kb.entities:
        if entity.has_pd_claim:
            assert len(entity.pd_association.citations) >= 1
            assert entity.pd_association.direction in {"up", "down", "variable"}


def test_citation_render_contains_pmid(kb):
    citation = next(iter(kb.citations.values()))
    assert f"PMID:{citation.pmid}" in citation.render()
    assert citation.url == f"https://pubmed.ncbi.nlm.nih.gov/{citation.pmid}/"


# -- lookup: hits ---------------------------------------------------------------

def test_lookup_resolves_affixed_gene_symbol(kb):
    assert kb.lookup("SNCA_expr", "transcriptomics").key == "SNCA"


def test_lookup_resolves_methylation_probe(kb):
    entity = kb.lookup("cg_SNCA", "epigenomics")
    assert entity is not None
    assert entity.key == "SNCA_INTRON1_METHYLATION"


def test_lookup_resolves_qiime_genus(kb):
    entity = kb.lookup("g__Akkermansia", "microbiome")
    assert entity is not None
    assert "Akkermansia" in entity.aliases


def test_lookup_resolves_rsid_to_gene(kb):
    assert kb.lookup("rs34637584").key == "LRRK2"


def test_lookup_ignores_case_and_punctuation(kb):
    assert kb.lookup("snca-EXPR").key == "SNCA"


# -- lookup: mandatory misses ---------------------------------------------------

def test_lookup_rejects_substring_matches(kb):
    # TGBA1P3 contains "GBA1" as a substring; whole-token matching must not
    # annotate it as glucocerebrosidase.
    assert kb.lookup("TGBA1P3") is None


def test_lookup_returns_none_for_uncurated_probes(kb):
    assert kb.lookup("1552281_a_at") is None
    assert kb.lookup("ENSG00000284733") is None


def test_lookup_respects_modality_filter(kb):
    entity = kb.lookup("Akkermansia", "microbiome")
    assert entity is not None
    if entity.modalities:
        assert kb.lookup("Akkermansia", "proteomics") is None


def test_lookup_empty_feature(kb):
    assert kb.lookup("") is None
    assert kb.lookup("___") is None


# -- loader validation ----------------------------------------------------------

def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


VALID_CITATIONS = """
GOOD_CIT:
  pmid: 12345
  title: A title
  journal: J
  year: 2020
  first_author: Doe J
"""


def test_loader_rejects_non_numeric_pmid(tmp_path):
    path = _write(tmp_path, "citations.yaml", """
BAD:
  pmid: PMC12345
  title: T
  journal: J
  year: 2020
  first_author: A
""")
    with pytest.raises(ValueError, match="non-numeric PMID"):
        _load_citations(path)


def test_loader_rejects_uncited_pd_claim(tmp_path):
    citations = _load_citations(_write(tmp_path, "citations.yaml", VALID_CITATIONS))
    entities = _write(tmp_path, "entities.yaml", """
- key: X
  aliases: [X]
  function: f
  pd_association:
    measured_as: expression
    direction: up
    statement: s
    citations: []
""")
    with pytest.raises(ValueError, match="no citations"):
        _load_entities(entities, citations)


def test_loader_rejects_unknown_citation_key(tmp_path):
    citations = _load_citations(_write(tmp_path, "citations.yaml", VALID_CITATIONS))
    entities = _write(tmp_path, "entities.yaml", """
- key: X
  aliases: [X]
  function: f
  pd_association:
    measured_as: expression
    direction: up
    statement: s
    citations: [NO_SUCH_KEY]
""")
    with pytest.raises(ValueError, match="unknown citation keys"):
        _load_entities(entities, citations)


def test_loader_rejects_invalid_direction(tmp_path):
    citations = _load_citations(_write(tmp_path, "citations.yaml", VALID_CITATIONS))
    entities = _write(tmp_path, "entities.yaml", """
- key: X
  aliases: [X]
  function: f
  pd_association:
    measured_as: expression
    direction: sideways
    statement: s
    citations: [GOOD_CIT]
""")
    with pytest.raises(ValueError, match="direction"):
        _load_entities(entities, citations)


def test_lookup_rejects_distinct_locus_suffixes(kb):
    # Antisense/divergent transcripts are different genes; SNCA-AS1 must not
    # inherit SNCA's PD annotation.
    assert kb.lookup("SNCA-AS1") is None
    assert kb.lookup("MAPT_AS1_expr") is None
    assert kb.lookup("GBA1-DT") is None
    assert kb.lookup("PINK1-AS") is None
