"""Re-verify every PMID in ``citations.yaml`` against PubMed.

Citations are the load-bearing part of this project's honesty claim, so they are
checkable rather than trusted. This module asks NCBI what each PMID actually
points at and compares the record title to the one we stored.

    python -m src.knowledge.verify_citations            # check all, exit 1 on drift
    python -m src.knowledge.verify_citations --quiet     # only report failures

Requires network access. The test suite skips it by default.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

from src.knowledge.kb import Citation, load_knowledge_base

ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
#: NCBI asks for no more than 3 requests/second without an API key.
REQUEST_INTERVAL_S = 0.4
#: Titles are compared after normalisation; PubMed occasionally reformats
#: punctuation or markup, so require high-but-not-exact similarity.
MIN_TITLE_RATIO = 0.90


@dataclass(frozen=True)
class VerificationResult:
    citation: Citation
    ok: bool
    detail: str

    def render(self) -> str:
        status = "OK  " if self.ok else "FAIL"
        return f"{status} {self.citation.key:26s} PMID:{self.citation.pmid:<9} {self.detail}"


def _normalise_title(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")
    text = text.lower().replace("α", "alpha").replace("β", "beta")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def fetch_summaries(pmids: list[str], timeout: int = 30) -> dict[str, dict]:
    """Fetch PubMed summary records for ``pmids`` (batched, rate-limited)."""
    records: dict[str, dict] = {}
    for start in range(0, len(pmids), 50):
        batch = pmids[start : start + 50]
        query = urllib.parse.urlencode({"db": "pubmed", "retmode": "json", "id": ",".join(batch)})
        with urllib.request.urlopen(f"{ESUMMARY}?{query}", timeout=timeout) as response:
            payload = json.load(response)
        result = payload.get("result", {})
        for pmid in batch:
            if pmid in result:
                records[pmid] = result[pmid]
        if start + 50 < len(pmids):
            time.sleep(REQUEST_INTERVAL_S)
    return records


def verify(citations: dict[str, Citation], timeout: int = 30) -> list[VerificationResult]:
    """Check that every stored PMID resolves to a record with the stored title."""
    ordered = sorted(citations.values(), key=lambda c: c.key)
    records = fetch_summaries([c.pmid for c in ordered], timeout=timeout)
    results = []
    for citation in ordered:
        record = records.get(citation.pmid)
        if record is None:
            results.append(VerificationResult(citation, False, "PMID not found in PubMed"))
            continue
        if record.get("error"):
            results.append(VerificationResult(citation, False, f"PubMed error: {record['error']}"))
            continue
        ratio = difflib.SequenceMatcher(
            None, _normalise_title(citation.title), _normalise_title(record.get("title", ""))
        ).ratio()
        if ratio < MIN_TITLE_RATIO:
            results.append(
                VerificationResult(
                    citation, False, f"title mismatch (ratio {ratio:.2f}): {record.get('title')!r}"
                )
            )
            continue
        results.append(
            VerificationResult(citation, True, f"{record.get('source')} {(record.get('pubdate') or '')[:4]}")
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify knowledge-base PMIDs against PubMed")
    parser.add_argument("--quiet", action="store_true", help="only print failures")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args(argv)

    kb = load_knowledge_base()
    results = verify(kb.citations, timeout=args.timeout)
    failures = [r for r in results if not r.ok]
    for result in results:
        if not args.quiet or not result.ok:
            print(result.render())
    print(f"\n{len(results) - len(failures)}/{len(results)} citations verified")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
