from __future__ import annotations
from functools import reduce
import pandas as pd
import numpy as np


class MOFAIntegrator:
    """
    Wraps mofapy2 for multi-omics factor analysis.

    MOFA+ decomposes multiple omics matrices into shared latent factors.
    Each factor captures co-variation across modalities; factors correlated
    with PD phenotype are the most biologically relevant.

    Note: fit_transform requires mofapy2 (pip install -e ".[training]").
    Subject alignment and utility methods work without it.
    """

    def __init__(self, n_factors: int = 20, max_iter: int = 1000,
                 convergence_mode: str = "fast"):
        self.n_factors = n_factors
        self.max_iter = max_iter
        self.convergence_mode = convergence_mode
        self._model = None

    def align_subjects(self, omics_dict: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Keep only subjects present in ALL modalities, sorted."""
        common = reduce(
            lambda a, b: a.intersection(b),
            [df.index for df in omics_dict.values()],
        )
        return {k: v.loc[sorted(common)] for k, v in omics_dict.items()}

    def fit_transform(self, omics_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Run MOFA+ and return factors matrix (subjects × n_factors)."""
        from mofapy2.run.entry_point import entry_point

        aligned = self.align_subjects(omics_dict)
        subjects = list(aligned[sorted(aligned.keys())[0]].index)

        ent = entry_point()
        data_list = [[aligned[m].values] for m in sorted(aligned.keys())]
        ent.set_data_options(scale_groups=False, scale_views=True)
        ent.set_data_matrix(
            data=data_list,
            likelihoods=["gaussian"] * len(aligned),
            views_names=sorted(aligned.keys()),
            groups_names=["group0"],
            samples_names=[subjects],
        )
        ent.set_model_options(factors=self.n_factors)
        ent.set_train_options(
            iter=self.max_iter,
            convergence_mode=self.convergence_mode,
            seed=42,
            verbose=False,
        )
        ent.build()
        ent.run()
        self._model = ent

        z = ent.model.nodes["Z"].getExpectation()
        cols = [f"factor_{i+1}" for i in range(self.n_factors)]
        return pd.DataFrame(z, index=subjects, columns=cols)

    def get_feature_weights(self, modality: str) -> pd.DataFrame:
        """Return feature loadings for a modality (requires prior fit_transform)."""
        if self._model is None:
            raise RuntimeError("Call fit_transform before get_feature_weights")
        w = self._model.model.nodes["W"].getExpectation()
        modalities = sorted(self._model.model.options["views_names"])
        idx = modalities.index(modality)
        cols = [f"factor_{i+1}" for i in range(self.n_factors)]
        return pd.DataFrame(w[idx], columns=cols)
