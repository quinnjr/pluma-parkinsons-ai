"""Metrics for generated Parkinson's-disease reports.

Exact-match accuracy on free-form text is always ~0 and tells you nothing, so
it is not used here. These metrics answer the questions that actually matter
for a grounded biomedical assistant:

* did it cite a PMID that exists in the curated knowledge base, or invent one?
* did it name the biomarkers the pipeline actually ranked?
* did it recover the model's diagnosis call?
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from src.knowledge import KnowledgeBase, load_knowledge_base

PMID_PATTERN = re.compile(r"PMID:\s*(\d+)", re.IGNORECASE)
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class CitationAudit:
    cited: tuple[str, ...]
    supported: tuple[str, ...]
    hallucinated: tuple[str, ...]

    @property
    def hallucination_rate(self) -> float:
        """Fraction of cited PMIDs that are not in the knowledge base."""
        return len(self.hallucinated) / len(self.cited) if self.cited else 0.0


def extract_pmids(text: str) -> list[str]:
    """PMIDs cited in a generated response, in order of first appearance."""
    seen: dict[str, None] = {}
    for match in PMID_PATTERN.finditer(text or ""):
        seen.setdefault(match.group(1), None)
    return list(seen)


def audit_citations(text: str, knowledge_base: KnowledgeBase | None = None) -> CitationAudit:
    """Check every cited PMID against the curated knowledge base.

    A PMID the knowledge base does not contain is not necessarily fake, but the
    model had no grounded reason to produce it, which is the failure this
    project exists to prevent.
    """
    kb = knowledge_base or load_knowledge_base()
    known = {c.pmid for c in kb.citations.values()}
    cited = tuple(extract_pmids(text))
    supported = tuple(p for p in cited if p in known)
    return CitationAudit(
        cited=cited,
        supported=supported,
        hallucinated=tuple(p for p in cited if p not in known),
    )


def _tokens(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall((text or "").lower())


def token_f1(prediction: str, reference: str) -> float:
    """Bag-of-tokens F1 between a prediction and its reference response."""
    pred_counts, ref_counts = Counter(_tokens(prediction)), Counter(_tokens(reference))
    overlap = sum((pred_counts & ref_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred_counts.values())
    recall = overlap / sum(ref_counts.values())
    return 2 * precision * recall / (precision + recall)


def biomarker_recall(prediction: str, expected_features: list[str]) -> float:
    """Fraction of the pipeline's ranked features the response actually names."""
    if not expected_features:
        return 0.0
    text = (prediction or "").lower()
    return sum(1 for f in expected_features if f.lower() in text) / len(expected_features)


def extract_diagnosis(text: str) -> str | None:
    """Recover the diagnosis call from a generated clinical-prediction response."""
    match = re.search(r"model output:\s*([A-Za-z]+)", text or "", re.IGNORECASE)
    if match:
        candidate = match.group(1).upper()
        if candidate in {"PD", "HC", "SWEDD", "PRODROMAL"}:
            return "Prodromal" if candidate == "PRODROMAL" else candidate
    return None


def compute_metrics(
    predictions: list[str],
    references: list[str],
    records: list[dict] | None = None,
    knowledge_base: KnowledgeBase | None = None,
) -> dict:
    """Aggregate generation metrics over a set of predictions.

    ``records`` are the source instruction pairs, used for their `grounding`
    block and task labels. Metrics that need them are omitted when absent
    rather than silently computed against nothing.
    """
    if not predictions:
        return {"n_samples": 0}
    if len(predictions) != len(references):
        raise ValueError(
            f"got {len(predictions)} predictions but {len(references)} references"
        )
    kb = knowledge_base or load_knowledge_base()
    records = records or [{} for _ in predictions]
    if len(records) != len(predictions):
        raise ValueError(f"got {len(records)} records but {len(predictions)} predictions")

    audits = [audit_citations(p, kb) for p in predictions]
    total_cited = sum(len(a.cited) for a in audits)
    total_hallucinated = sum(len(a.hallucinated) for a in audits)
    metrics = {
        "n_samples": len(predictions),
        "token_f1": _mean(token_f1(p, r) for p, r in zip(predictions, references)),
        "citation_hallucination_rate": _mean(a.hallucination_rate for a in audits),
        # The macro rate above scores a response that cites nothing as a perfect
        # 0.0, so a model that stops citing gets a perfect score. The micro rate
        # over all cited PMIDs is not gameable that way; read the two together
        # with responses_with_citations.
        "citation_hallucination_rate_micro": (
            total_hallucinated / total_cited if total_cited else 0.0
        ),
        "responses_with_citations": _mean(1.0 if a.cited else 0.0 for a in audits),
    }

    expected = [(record.get("grounding") or {}).get("annotated_features") for record in records]
    # Only score recall where there was something to recall; averaging in 0.0
    # for profiles with no annotated features reads as failure where none exists.
    scorable = [(p, e) for p, e in zip(predictions, expected) if e]
    if scorable:
        metrics["biomarker_recall"] = _mean(biomarker_recall(p, e) for p, e in scorable)

    diagnosis_pairs = [
        (extract_diagnosis(p), extract_diagnosis(r))
        for p, r, record in zip(predictions, references, records)
        if record.get("task") == "clinical_prediction"
    ]
    scored = [(p, r) for p, r in diagnosis_pairs if r is not None]
    if scored:
        metrics["diagnosis_accuracy"] = _mean(1.0 if p == r else 0.0 for p, r in scored)
        metrics["diagnosis_parse_rate"] = _mean(
            1.0 if p is not None else 0.0 for p, _ in scored
        )
    return metrics


def _mean(values) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0
