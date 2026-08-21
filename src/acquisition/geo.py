from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.utils import ensure_dir

if TYPE_CHECKING:
    import pandas as pd

PD_KEYWORDS = [
    "parkinson", "parkinson's", "pd ",
    "dopaminergic", "substantia nigra",
    "lewy body", "alpha-synuclein",
]

GEO_FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series"


@dataclass
class GEOStudy:
    accession: str
    title: str
    organism: str
    platform: str
    n_samples: int


class GEOClient:
    """Downloads and parses GEO datasets for PD multi-omics analysis."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        ensure_dir(self.data_dir)

    def _build_soft_url(self, accession: str) -> str:
        prefix = accession[:6] + "nnn"
        return f"{GEO_FTP_BASE}/{prefix}/{accession}/soft/{accession}_family.soft.gz"

    def filter_pd_studies(self, studies: list[GEOStudy]) -> list[GEOStudy]:
        """Keep only studies whose title contains PD-relevant keywords."""
        return [
            s for s in studies
            if any(kw in s.title.lower() for kw in PD_KEYWORDS)
        ]

    def download_study(self, accession: str) -> Path:
        """Download a GEO SOFT file. Returns local directory path."""
        import GEOparse
        dest = self.data_dir / accession
        ensure_dir(dest)
        GEOparse.get_GEO(geo=accession, destdir=str(dest), silent=True)
        return dest

    def parse_expression_matrix(self, accession: str) -> pd.DataFrame:
        """Parse downloaded SOFT file into a genes x samples expression matrix."""
        import GEOparse
        import pandas as pd
        dest = self.data_dir / accession
        gse = GEOparse.get_GEO(geo=accession, destdir=str(dest), silent=True)
        frames = []
        for gsm_name, gsm in gse.gsms.items():
            if gsm.table is not None and not gsm.table.empty:
                col = gsm.table.set_index("ID_REF")["VALUE"].rename(gsm_name)
                frames.append(col)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).apply(pd.to_numeric, errors="coerce")

    #: Column names used by GEO platform annotation tables for the gene symbol.
    SYMBOL_COLUMNS = ("Gene Symbol", "GENE_SYMBOL", "gene_symbol", "Symbol", "GENE")

    def probe_to_gene(self, accession: str) -> dict[str, str]:
        """Map platform probe IDs to HGNC-style gene symbols.

        Without this, a microarray series yields features like ``1007_s_at``,
        which no knowledge base can annotate. Probes mapping to several genes
        are dropped rather than arbitrarily assigned to the first one.
        """
        import GEOparse

        gse = GEOparse.get_GEO(geo=accession, destdir=str(self.data_dir / accession), silent=True)
        mapping: dict[str, str] = {}
        for gpl in gse.gpls.values():
            table = getattr(gpl, "table", None)
            if table is None or table.empty:
                continue
            symbol_col = next((c for c in self.SYMBOL_COLUMNS if c in table.columns), None)
            if symbol_col is None:
                continue
            id_col = "ID" if "ID" in table.columns else table.columns[0]
            for probe, symbol in zip(table[id_col], table[symbol_col]):
                if not isinstance(symbol, str):
                    continue
                symbol = symbol.strip()
                # "GENE1 /// GENE2" means the probe cannot distinguish them.
                if not symbol or "///" in symbol:
                    continue
                mapping[str(probe)] = symbol
        return mapping

    @staticmethod
    def collapse_to_genes(expr: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
        """Collapse a probe x sample matrix to gene x sample, keeping the most
        variable probe per gene.

        Averaging probes for the same gene mixes probes with different
        hybridisation behaviour; taking the most variable one is the
        conventional choice and keeps each row traceable to a single probe.
        """
        import pandas as pd

        if not expr.index.is_unique:
            # .loc on a duplicated label returns every matching row, which
            # would misalign the probe->symbol pairing below.
            expr = expr[~expr.index.duplicated(keep="first")]
        symbols = pd.Series({p: mapping[p] for p in expr.index if p in mapping})
        if symbols.empty:
            return expr.iloc[0:0]
        annotated = expr.loc[symbols.index]
        order = annotated.var(axis=1).sort_values(ascending=False).index
        ranked = annotated.loc[order]
        ranked_symbols = symbols.loc[order]
        keep = ~ranked_symbols.duplicated()
        collapsed = ranked.loc[keep]
        collapsed.index = pd.Index(ranked_symbols.loc[keep].values, name="gene_symbol")
        return collapsed.sort_index()
