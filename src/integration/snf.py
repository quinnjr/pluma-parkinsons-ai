from __future__ import annotations
import pandas as pd
import numpy as np
from src.utils import ensure_dir


class SNFIntegrator:
    """
    Fuses patient similarity networks across modalities via SNF.
    Requires snfpy: pip install snfpy

    SNF complements MOFA+ with a non-linear approach: builds per-modality
    patient affinity graphs and iteratively diffuses information between them.
    The fused graph captures patient subgroups that no single modality reveals.
    """

    def __init__(self, k_neighbors: int = 20, mu: float = 0.5, t_iterations: int = 20):
        self.k_neighbors = k_neighbors
        self.mu = mu
        self.t_iterations = t_iterations

    def fit_transform(self, omics_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Fuse affinity matrices. Returns subjects × subjects fused matrix."""
        import snf

        modalities = sorted(omics_dict.keys())
        subjects = list(omics_dict[modalities[0]].index)

        affinity_matrices = []
        for m in modalities:
            df = omics_dict[m].loc[subjects]
            aff = snf.make_affinity(df.values, metric="euclidean",
                                    K=self.k_neighbors, mu=self.mu)
            affinity_matrices.append(aff)

        fused = snf.snf(affinity_matrices, K=self.k_neighbors, t=self.t_iterations)
        return pd.DataFrame(fused, index=subjects, columns=subjects)
