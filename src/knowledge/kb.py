"""Curated Parkinson's disease knowledge base.

The point of this module is to make the difference between "we measured this"
and "the literature says this" mechanical rather than rhetorical. A feature that
survives SHAP ranking is a *statistical* observation; attaching a mechanism to
it is a *literature* claim, and a literature claim without a citation is a
fabrication. `lookup` returns `None` for anything not curated, and callers are
expected to say so rather than improvise.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

DATA_DIR = Path(__file__).parent
CITATIONS_PATH = DATA_DIR / "citations.yaml"
ENTITIES_PATH = DATA_DIR / "entities.yaml"

#: Affixes that appear in feature names purely as bookkeeping and carry no
#: identity of their own (``SNCA_expr`` and ``SNCA`` are the same gene).
_AFFIXES = (
    "expr", "exprs", "expression", "rna", "mrna", "transcript",
    "prot", "protein", "csf", "serum", "plasma", "blood",
    "cpg", "cg", "meth", "methylation", "beta",
    "bug", "otu", "asv", "genus", "species", "abundance",
    "metab", "metabolite", "snp", "variant", "allele", "dosage",
    "g", "f", "p", "c", "o", "s", "k",  # QIIME-style rank prefixes: g__, f__, ...
)

_VALID_DIRECTIONS = {"up", "down", "variable"}

#: HGNC suffixes that turn a gene symbol into a *different* gene: antisense
#: transcripts (-AS1), divergent transcripts (-DT), intronic transcripts (-IT1),
#: opposite-strand (-OS). ``SNCA-AS1`` must not inherit SNCA's PD annotation.
_DISTINCT_LOCUS_SUFFIXES = {"as", "as1", "as2", "as3", "dt", "it1", "os"}


@dataclass(frozen=True)
class Citation:
    key: str
    pmid: str
    title: str
    journal: str
    year: int
    first_author: str

    def render(self) -> str:
        """Short inline form, e.g. ``Sidransky E, N Engl J Med 2009, PMID:19846850``."""
        return f"{self.first_author}, {self.journal} {self.year}, PMID:{self.pmid}"

    @property
    def url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"


@dataclass(frozen=True)
class PDAssociation:
    """A cited claim about an entity's behaviour in Parkinson's disease."""

    measured_as: str
    direction: str
    statement: str
    citations: tuple[Citation, ...]

    def render_citations(self) -> str:
        return "; ".join(c.render() for c in self.citations)


@dataclass(frozen=True)
class Entity:
    key: str
    aliases: tuple[str, ...]
    modalities: tuple[str, ...]
    function: str
    pd_association: PDAssociation | None = None

    @property
    def has_pd_claim(self) -> bool:
        return self.pd_association is not None


def _normalise(text: str) -> str:
    """Casefold and collapse everything that is not alphanumeric to single spaces."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _tokens(text: str) -> list[str]:
    """Identity-bearing tokens of a feature name, with bookkeeping affixes dropped."""
    return [t for t in _normalise(text).split() if t and t not in _AFFIXES]


@dataclass
class KnowledgeBase:
    entities: tuple[Entity, ...]
    citations: dict[str, Citation] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Longest aliases first so "alpha synuclein" wins over a bare "snca"
        # substring of some longer probe name, and so multi-word taxa match
        # before their genus.
        index: list[tuple[list[str], Entity]] = []
        for entity in self.entities:
            for alias in entity.aliases:
                alias_tokens = _tokens(alias)
                if alias_tokens:
                    index.append((alias_tokens, entity))
        self._alias_index = sorted(index, key=lambda pair: -len(" ".join(pair[0])))
        self._by_key = {e.key: e for e in self.entities}

    # -- access ----------------------------------------------------------------

    def get(self, key: str) -> Entity | None:
        return self._by_key.get(key)

    def citation(self, key: str) -> Citation | None:
        return self.citations.get(key)

    def lookup(self, feature: str, modality: str | None = None) -> Entity | None:
        """Resolve a feature name to a curated entity, or ``None`` if uncurated.

        Matching is deliberately conservative: an alias must appear as a whole
        token (or contiguous run of tokens) in the feature name. Substring
        matching would map ``TGBA1P3`` onto ``GBA1``, which is exactly the sort
        of confident-but-wrong annotation this module exists to prevent.
        """
        haystack = _tokens(feature)
        if not haystack:
            return None
        for alias_tokens, entity in self._alias_index:
            if not _matches(haystack, alias_tokens):
                continue
            if modality and entity.modalities and modality not in entity.modalities:
                continue
            return entity
        return None

    def modality_entities(self, modality: str) -> list[Entity]:
        return [e for e in self.entities if modality in e.modalities]


def _matches(haystack: list[str], needle: list[str]) -> bool:
    """True if ``needle`` appears as a contiguous token run inside ``haystack``,
    and the run is not immediately followed by a distinct-locus suffix."""
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] != needle:
            continue
        following = haystack[i + n] if i + n < len(haystack) else None
        if following in _DISTINCT_LOCUS_SUFFIXES:
            continue
        return True
    return False


def _load_citations(path: Path) -> dict[str, Citation]:
    raw = yaml.safe_load(path.read_text()) or {}
    citations: dict[str, Citation] = {}
    for key, value in raw.items():
        pmid = str(value["pmid"])
        if not pmid.isdigit():
            raise ValueError(f"citation {key!r} has a non-numeric PMID: {pmid!r}")
        citations[key] = Citation(
            key=key,
            pmid=pmid,
            title=value["title"],
            journal=value["journal"],
            year=int(value["year"]),
            first_author=value["first_author"],
        )
    return citations


def _load_entities(path: Path, citations: dict[str, Citation]) -> tuple[Entity, ...]:
    raw = yaml.safe_load(path.read_text()) or []
    entities = []
    for item in raw:
        assoc = None
        if "pd_association" in item:
            spec = item["pd_association"]
            direction = spec["direction"]
            if direction not in _VALID_DIRECTIONS:
                raise ValueError(
                    f"entity {item['key']!r}: direction must be one of "
                    f"{sorted(_VALID_DIRECTIONS)}, got {direction!r}"
                )
            keys = spec.get("citations") or []
            if not keys:
                raise ValueError(
                    f"entity {item['key']!r} has a pd_association with no citations. "
                    "Uncited PD claims are not allowed; drop the association instead."
                )
            missing = [k for k in keys if k not in citations]
            if missing:
                raise ValueError(
                    f"entity {item['key']!r} cites unknown citation keys: {missing}"
                )
            assoc = PDAssociation(
                measured_as=spec["measured_as"],
                direction=direction,
                statement=" ".join(spec["statement"].split()),
                citations=tuple(citations[k] for k in keys),
            )
        entities.append(
            Entity(
                key=item["key"],
                aliases=tuple(item["aliases"]),
                modalities=tuple(item.get("modalities", ())),
                function=" ".join(item["function"].split()),
                pd_association=assoc,
            )
        )
    return tuple(entities)


@lru_cache(maxsize=1)
def load_knowledge_base(
    citations_path: str | Path = CITATIONS_PATH,
    entities_path: str | Path = ENTITIES_PATH,
) -> KnowledgeBase:
    """Load and validate the bundled knowledge base (cached)."""
    citations = _load_citations(Path(citations_path))
    entities = _load_entities(Path(entities_path), citations)
    return KnowledgeBase(entities=entities, citations=citations)
