from __future__ import annotations
import pandas as pd

DEFAULT_PD_GENES = ["LRRK2", "GBA", "SNCA", "PINK1", "PRKN", "PARK7", "VPS35", "CHCHD2"]


class GenomicsPreprocessor:
    def __init__(self, maf_threshold: float = 0.01,
                 known_pd_genes: list[str] | None = None):
        self.maf_threshold = maf_threshold
        self.known_pd_genes = set(known_pd_genes) if known_pd_genes is not None else set(DEFAULT_PD_GENES)

    def filter_by_maf(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rare variants unless they are in known PD causal genes."""
        is_pd_gene = df["gene"].isin(self.known_pd_genes)
        passes_maf = df["maf"] >= self.maf_threshold
        return df[passes_maf | is_pd_gene].copy()

    def encode_dosage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure genotype is 0/1/2 dosage encoding."""
        df = df.copy()
        df["genotype"] = df["genotype"].clip(0, 2).astype(int)
        return df

    def pivot_to_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pivot long-format variant table to subjects x variants matrix."""
        return df.pivot_table(
            index="subject_id",
            columns="variant_id",
            values="genotype",
            aggfunc="first",
            fill_value=0,
        )

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.filter_by_maf(df)
        df = self.encode_dosage(df)
        return self.pivot_to_feature_matrix(df)
