"""Simulated omics modalities for cohorts where the real assay is unavailable.

PPMI requires registration and approval, so a pipeline run that has only public
GEO and NHANES data still needs *something* in the remaining modalities to
exercise integration end to end. This module supplies it, under three rules:

1. Every modality produced here is recorded in
   :class:`~src.models.Provenance.synthetic_modalities`, and that flag reaches
   the generated training text as an explicit caveat.
2. Feature names are the real identifiers the corresponding assay would produce
   (``SNCA``, ``Prevotella``, ``urate``), so downstream annotation and modality
   inference are exercised the way they will be on real data.
3. The PD-versus-control shifts are applied in the direction the curated
   literature reports, so a run on simulated data is a test of the *plumbing*
   and produces internally consistent text. It is not evidence about
   Parkinson's disease, and nothing in this repository treats it as such.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimulatedFeature:
    """One feature, with the effect direction taken from the curated literature."""

    name: str
    #: Multiplicative or additive shift applied to PD subjects. Positive means
    #: higher in PD, matching the `direction` recorded in the knowledge base.
    pd_shift: float


# Directions here mirror src/knowledge/entities.yaml. Where the knowledge base
# records "variable", no shift is applied.
GENOMICS_RISK_VARIANTS = (
    SimulatedFeature("rs34637584", 0.15),   # LRRK2 G2019S
    SimulatedFeature("rs76763715", 0.08),   # GBA1 N370S
    SimulatedFeature("rs356181", 0.12),     # SNCA
    SimulatedFeature("rs393152", 0.10),     # MAPT
    SimulatedFeature("rs188286943", 0.02),  # VPS35 D620N
)

EPIGENOMICS_CPGS = (
    SimulatedFeature("cg_SNCA_intron1", -0.15),  # hypomethylated in PD
)

PROTEOMICS_TARGETS = (
    SimulatedFeature("csf_SNCA", -0.30),  # reduced CSF alpha-synuclein
    SimulatedFeature("csf_NfL", 0.25),    # raised with axonal injury
    SimulatedFeature("csf_MAPT", 0.10),
)

METABOLOMICS_TARGETS = (
    SimulatedFeature("metab_urate", -0.30),
    SimulatedFeature("metab_caffeine", -0.35),
    SimulatedFeature("metab_paraxanthine", -0.30),
)

MICROBIOME_TAXA = (
    SimulatedFeature("bug_Prevotella", -0.40),
    SimulatedFeature("bug_Akkermansia", 0.30),
    SimulatedFeature("bug_Enterobacteriaceae", 0.25),
    SimulatedFeature("bug_Faecalibacterium", -0.25),
    SimulatedFeature("bug_Roseburia", -0.25),
    SimulatedFeature("bug_Bifidobacterium", 0.20),
)


def _frame(data: np.ndarray, names: list[str], subjects: list[str]) -> pd.DataFrame:
    return pd.DataFrame(data, index=pd.Index(subjects, name="subject_id"), columns=names)


def _padded_names(curated: tuple[SimulatedFeature, ...], total: int, prefix: str) -> list[str]:
    """Curated identifiers first, then uncurated filler features.

    The filler is not padding for its own sake: a real assay measures far more
    features than any knowledge base annotates, and the formatter's abstention
    path only gets exercised if some top-ranked features are unannotated.
    """
    names = [f.name for f in curated]
    names += [f"{prefix}_unannotated_{i}" for i in range(total - len(names))]
    return names


class SyntheticModalityGenerator:
    """Generate simulated omics matrices with literature-directed PD shifts."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate(self, subjects: list[str], diagnosis: pd.Series,
                 modalities: tuple[str, ...]) -> dict[str, pd.DataFrame]:
        logger.warning(
            "Generating SIMULATED data for %s. These values are not measurements and "
            "must not be interpreted as evidence about Parkinson's disease.",
            ", ".join(modalities),
        )
        is_pd = (diagnosis.loc[subjects] == "PD").to_numpy().astype(float)
        builders = {
            "genomics": self._genomics,
            "epigenomics": self._epigenomics,
            "proteomics": self._proteomics,
            "metabolomics": self._metabolomics,
            "microbiome": self._microbiome,
        }
        unknown = [m for m in modalities if m not in builders]
        if unknown:
            raise ValueError(
                f"No simulator for modality/modalities {unknown}. Available: {sorted(builders)}"
            )
        return {m: builders[m](subjects, is_pd) for m in modalities}

    # -- per-modality simulators ----------------------------------------------

    def _genomics(self, subjects: list[str], is_pd: np.ndarray, n_total: int = 30) -> pd.DataFrame:
        names = _padded_names(GENOMICS_RISK_VARIANTS, n_total, "snp")
        data = self.rng.integers(0, 3, size=(len(subjects), n_total)).astype(float)
        for i, feature in enumerate(GENOMICS_RISK_VARIANTS):
            # Risk-allele carriage: baseline population frequency plus a PD excess.
            data[:, i] = self.rng.binomial(1, np.clip(0.05 + feature.pd_shift * is_pd, 0, 1))
        return _frame(data, names, subjects)

    def _epigenomics(self, subjects: list[str], is_pd: np.ndarray, n_total: int = 40) -> pd.DataFrame:
        names = _padded_names(EPIGENOMICS_CPGS, n_total, "cg")
        data = self.rng.uniform(0.1, 0.9, size=(len(subjects), n_total))
        for i, feature in enumerate(EPIGENOMICS_CPGS):
            data[:, i] = np.clip(data[:, i] + feature.pd_shift * is_pd, 0.01, 0.99)
        return _frame(data, names, subjects)

    def _proteomics(self, subjects: list[str], is_pd: np.ndarray, n_total: int = 25) -> pd.DataFrame:
        names = _padded_names(PROTEOMICS_TARGETS, n_total, "prot")
        data = self.rng.lognormal(0, 0.5, size=(len(subjects), n_total))
        for i, feature in enumerate(PROTEOMICS_TARGETS):
            data[:, i] *= 1 + feature.pd_shift * is_pd
        return _frame(data, names, subjects)

    def _metabolomics(self, subjects: list[str], is_pd: np.ndarray, n_total: int = 35) -> pd.DataFrame:
        names = _padded_names(METABOLOMICS_TARGETS, n_total, "metab")
        data = self.rng.exponential(1.0, size=(len(subjects), n_total)) + 0.1
        for i, feature in enumerate(METABOLOMICS_TARGETS):
            data[:, i] *= 1 + feature.pd_shift * is_pd
        return _frame(data, names, subjects)

    def _microbiome(self, subjects: list[str], is_pd: np.ndarray, n_total: int = 20) -> pd.DataFrame:
        names = _padded_names(MICROBIOME_TAXA, n_total, "bug")
        data = self.rng.dirichlet(np.ones(n_total), size=len(subjects)) * 10_000
        for i, feature in enumerate(MICROBIOME_TAXA):
            data[:, i] *= 1 + feature.pd_shift * is_pd
        return _frame(data, names, subjects)
