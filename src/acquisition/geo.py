from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from src.utils import ensure_dir

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

    def parse_expression_matrix(self, accession: str) -> "pd.DataFrame":
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
