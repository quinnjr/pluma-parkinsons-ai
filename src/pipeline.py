"""Top-level pipeline orchestrator for the PD multi-omics training pipeline.

Each stage reads its inputs from disk and writes its outputs back, so stages can
run on different machines at different times:

    python -m src.pipeline --stage acquire            # network, CPU
    python -m src.pipeline --stage preprocess
    python -m src.pipeline --stage integrate
    python -m src.pipeline --stage build_instructions
    python -m src.pipeline --stage train              # GPU
    python -m src.pipeline --stage all

``--data-root`` rebases every path, which is how the smoke test runs the real
pipeline against a small cohort without touching the main data directories.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import Provenance, Stage1Output
from src.utils import ensure_dir, load_config

logger = logging.getLogger(__name__)

STAGES = ["acquire", "preprocess", "integrate", "build_instructions", "train"]

#: Written by `acquire`, read by `preprocess`; keeps subject-level metadata that
#: is not part of any single modality matrix.
MANIFEST_FILENAME = "manifest.csv"


class Pipeline:
    def __init__(
        self,
        acquisition_config: str = "configs/acquisition.yaml",
        preprocessing_config: str = "configs/preprocessing.yaml",
        integration_config: str = "configs/integration.yaml",
        training_config: str = "configs/training.yaml",
        data_root: str | Path | None = None,
        max_subjects: int | None = None,
    ):
        self.acq_cfg = load_config(acquisition_config)
        self.pre_cfg = load_config(preprocessing_config)
        self.int_cfg = load_config(integration_config)
        self.training_config_path = training_config
        self.train_cfg = load_config(training_config)

        self.cohort = dict(self.acq_cfg.get("cohort", {}))
        if max_subjects is not None:
            self.cohort["max_subjects"] = max_subjects

        paths = self.acq_cfg.get("paths", {})
        if data_root is not None:
            root = Path(data_root)
            paths = {key: str(root / key) for key in ("raw", "processed", "integrated", "instructions")}
        self.paths = {key: Path(value) for key, value in paths.items()}

    # -- paths -----------------------------------------------------------------

    def path(self, key: str) -> Path:
        try:
            return self.paths[key]
        except KeyError:
            raise KeyError(
                f"No '{key}' path configured. Add it under `paths:` in the acquisition "
                f"config, or pass --data-root."
            ) from None

    def run_stage(self, stage: str) -> None:
        if stage == "all":
            for s in STAGES:
                logger.info("=== stage: %s ===", s)
                self.run_stage(s)
            return
        if stage not in STAGES:
            raise ValueError(f"Unknown stage {stage!r}. Choose from: {STAGES}")
        getattr(self, f"_run_{stage}")()

    # -- stage 1: acquire ------------------------------------------------------

    def _run_acquire(self) -> None:
        """Download the real public data and record the subject manifest."""
        from src.acquisition.environmental import NHANESClient
        from src.acquisition.geo import GEOClient

        raw = ensure_dir(self.path("raw"))
        accession = self.cohort["geo_accession"]
        max_subjects = self.cohort.get("max_subjects")

        logger.info("Downloading GEO series %s", accession)
        geo = GEOClient(raw / "geo")
        expr = geo.parse_expression_matrix(accession)
        diagnosis = self._geo_diagnoses(accession, raw / "geo")

        subjects = [s for s in expr.columns if diagnosis.get(s) in {"PD", "HC"}]
        if max_subjects:
            subjects = subjects[:max_subjects]
        if not subjects:
            raise RuntimeError(
                f"{accession} yielded no samples with a PD/HC label; check the series "
                f"characteristics fields."
            )
        expr = expr[subjects]

        mapping = geo.probe_to_gene(accession)
        genes = geo.collapse_to_genes(expr, mapping)
        logger.info(
            "%s: %d probes -> %d gene symbols across %d subjects",
            accession, len(expr), len(genes), len(subjects),
        )
        if genes.empty:
            raise RuntimeError(
                f"No probe could be mapped to a gene symbol for {accession}. Downstream "
                f"annotation would be impossible, so this is a hard failure rather than "
                f"a silent fallback to probe IDs."
            )
        genes.to_csv(raw / "transcriptomics_counts.csv")

        env_scores = self._nhanes_risk_scores(NHANESClient(raw / "nhanes"), subjects)

        manifest = pd.DataFrame(
            {
                "subject_id": subjects,
                "diagnosis": [diagnosis[s] for s in subjects],
                # The GEO series carries no staging information. Leaving this
                # empty is the honest option; inventing "early" for every PD
                # subject would put a fabricated label into training targets.
                "disease_stage": [None] * len(subjects),
                "environmental_risk_score": [env_scores.get(s, 0.0) for s in subjects],
            }
        )
        manifest.to_csv(raw / MANIFEST_FILENAME, index=False)
        logger.info("Manifest written to %s (%d subjects)", raw / MANIFEST_FILENAME, len(manifest))

    @staticmethod
    def _geo_diagnoses(accession: str, destdir: Path) -> dict[str, str]:
        """Read PD/HC labels out of each sample's characteristics fields."""
        import GEOparse

        gse = GEOparse.get_GEO(geo=accession, destdir=str(destdir / accession), silent=True)
        labels = {}
        for name, gsm in gse.gsms.items():
            characteristics = " ".join(gsm.metadata.get("characteristics_ch1", [])).lower()
            if "parkinson" in characteristics:
                labels[name] = "PD"
            elif any(term in characteristics for term in ("control", "healthy", "normal")):
                labels[name] = "HC"
        return labels

    def _nhanes_risk_scores(self, client, subjects: list[str]) -> dict[str, float]:
        """Scale real NHANES blood-lead values onto a 0-10 exposure score.

        NHANES participants are not the GEO subjects; this attaches a realistic
        exposure *distribution* to the cohort, which is why the environmental
        modality is reported as simulated in the provenance record.
        """
        cycle = self.cohort.get("nhanes_cycle", "2017-2018")
        column = self.cohort.get("nhanes_metals_column", "LBXBPB")
        path = client.download_file(cycle, "metals")
        frame = pd.read_sas(str(path), format="xport", encoding="utf-8")
        if column not in frame.columns:
            raise RuntimeError(
                f"NHANES file {path.name} has no column {column!r}; available: "
                f"{list(frame.columns)[:20]}"
            )
        values = frame[column].dropna().to_numpy()
        if values.size == 0:
            raise RuntimeError(f"NHANES column {column!r} is entirely missing in {path.name}")
        spread = values.max() - values.min()
        scaled = (values - values.min()) / spread * 10 if spread > 0 else np.full_like(values, 5.0)
        # Deterministic assignment: subject i takes the i-th NHANES value,
        # wrapping if the cohort is larger than the NHANES sample.
        return {s: float(scaled[i % len(scaled)]) for i, s in enumerate(subjects)}

    # -- stage 2: preprocess ---------------------------------------------------

    def _run_preprocess(self) -> None:
        from src.acquisition.synthetic import SyntheticModalityGenerator
        from src.preprocessing.epigenomics import EpigenomicsPreprocessor
        from src.preprocessing.metabolomics import MetabolomicsPreprocessor
        from src.preprocessing.microbiome import MicrobiomePreprocessor
        from src.preprocessing.proteomics import ProteomicsPreprocessor
        from src.preprocessing.transcriptomics import TranscriptomicsPreprocessor

        raw, processed = self.path("raw"), ensure_dir(self.path("processed"))
        manifest = pd.read_csv(raw / MANIFEST_FILENAME)
        subjects = manifest["subject_id"].astype(str).tolist()
        diagnosis = pd.Series(manifest["diagnosis"].values, index=subjects)

        counts = pd.read_csv(raw / "transcriptomics_counts.csv", index_col=0)
        counts.columns = [str(c) for c in counts.columns]
        counts = counts[subjects]

        txn_cfg = self.pre_cfg["transcriptomics"]
        preprocessor = TranscriptomicsPreprocessor(
            min_count=txn_cfg["min_count"],
            min_sample_fraction=txn_cfg["min_samples_expressed_fraction"],
        )
        # Array intensities have no gene lengths; TPM's length term is a no-op
        # for them, so use a constant and let the per-sample scaling do the work.
        gene_lengths = pd.Series(1000.0, index=counts.index)
        transcriptomics = preprocessor.preprocess(counts, gene_lengths).T
        top_n = self.cohort.get("transcriptomics_top_variable_genes")
        if top_n and transcriptomics.shape[1] > top_n:
            keep = transcriptomics.var().nlargest(top_n).index
            transcriptomics = transcriptomics.loc[:, keep]
        matrices = {"transcriptomics": transcriptomics}

        simulated = tuple(self.cohort.get("simulated_modalities", ()))
        if simulated:
            raw_simulated = SyntheticModalityGenerator(
                seed=self.cohort.get("seed", 42)
            ).generate(subjects, diagnosis, simulated)
            preprocessors = {
                "epigenomics": lambda df: EpigenomicsPreprocessor().preprocess(df),
                "proteomics": lambda df: ProteomicsPreprocessor().preprocess(df),
                "metabolomics": lambda df: MetabolomicsPreprocessor().preprocess(df),
                "microbiome": lambda df: MicrobiomePreprocessor(
                    rarefaction_depth=self.pre_cfg["microbiome"]["rarefaction_depth"]
                ).preprocess(df),
                # Genotype dosages are already model-ready.
                "genomics": lambda df: df.clip(0, 2),
            }
            for modality, frame in raw_simulated.items():
                matrices[modality] = preprocessors[modality](frame)

        for modality, frame in matrices.items():
            frame.to_csv(processed / f"{modality}.csv")
            logger.info("  %-16s %s -> %s", modality, tuple(frame.shape), processed / f"{modality}.csv")
        manifest.to_csv(processed / MANIFEST_FILENAME, index=False)

    # -- stage 3: integrate ----------------------------------------------------

    def _run_integrate(self) -> None:
        from src.integration.ensemble import OmicsEnsemble
        from src.integration.mofa import MOFAIntegrator
        from src.integration.stage1_builder import Stage1Builder

        processed = self.path("processed")
        integrated = ensure_dir(self.path("integrated"))
        manifest = pd.read_csv(processed / MANIFEST_FILENAME)
        subjects = manifest["subject_id"].astype(str).tolist()

        matrices = {}
        for path in sorted(processed.glob("*.csv")):
            if path.name == MANIFEST_FILENAME:
                continue
            frame = pd.read_csv(path, index_col=0)
            frame.index = [str(i) for i in frame.index]
            matrices[path.stem] = frame.loc[subjects]
        if not matrices:
            raise RuntimeError(f"No processed modality matrices found in {processed}")
        logger.info("Integrating %d modalities: %s", len(matrices), ", ".join(sorted(matrices)))

        mofa_cfg = self.int_cfg["mofa"]
        n_factors = min(mofa_cfg["n_factors"], max(2, len(subjects) // 2))
        factors = MOFAIntegrator(
            n_factors=n_factors,
            max_iter=mofa_cfg["max_iter"],
            convergence_mode=mofa_cfg["convergence_mode"],
        ).fit_transform(matrices)

        # Feature names are namespaced by modality so downstream inference never
        # has to guess where a column came from.
        blocks = [
            frame.rename(columns=lambda c, m=modality: f"{m}:{c}")
            for modality, frame in sorted(matrices.items())
        ]
        X = pd.concat(blocks + [factors.rename(columns=lambda c: f"integrated:{c}")], axis=1)
        X.columns = [str(c) for c in X.columns]
        y = pd.Series(manifest["diagnosis"].values, index=subjects)

        ensemble_cfg = self.int_cfg["ensemble"]
        ensemble = OmicsEnsemble(
            n_estimators=ensemble_cfg.get("n_estimators", 200),
            max_depth=ensemble_cfg.get("max_depth", 6),
            learning_rate=ensemble_cfg.get("learning_rate", 0.05),
        )
        fit = ensemble.fit_evaluate(X, y, n_splits=ensemble_cfg["cv_folds"])
        logger.info(
            "Ensemble: out_of_fold=%s folds=%d cv_auc=%s",
            fit.out_of_fold, fit.n_splits,
            "n/a" if fit.cv_auc is None else f"{fit.cv_auc:.3f}",
        )

        environmental = pd.Series(
            manifest["environmental_risk_score"].values, index=subjects
        )
        stages = pd.Series(manifest["disease_stage"].values, index=subjects).where(
            manifest["disease_stage"].notna().values, None
        )
        provenance = Provenance(
            cohort_size=len(subjects),
            synthetic_modalities=tuple(self.cohort.get("simulated_modalities", ()))
            + ("environmental",),
            datasets=tuple(self.cohort.get("datasets", ()))
            or (self.cohort["geo_accession"], f"NHANES {self.cohort.get('nhanes_cycle', '')}".strip()),
            model=f"XGBoost + TreeSHAP over {X.shape[1]} features from {len(matrices)} modalities",
        )
        outputs = Stage1Builder(
            top_k_biomarkers=ensemble_cfg["top_k_biomarkers"]
        ).build(X, y, fit, factors, environmental, stages, provenance)

        for output in outputs:
            (integrated / f"{output.subject_id}.json").write_text(output.to_json())
        logger.info("Wrote %d Stage1Output records to %s", len(outputs), integrated)

    # -- stage 4: build instructions -------------------------------------------

    def _run_build_instructions(self) -> None:
        from src.instruction_builder.dataset_builder import DatasetBuilder

        integrated = self.path("integrated")
        instructions = ensure_dir(self.path("instructions"))
        files = sorted(integrated.glob("*.json"))
        if not files:
            raise RuntimeError(f"No Stage1Output JSON files in {integrated}; run --stage integrate")

        outputs = [Stage1Output.from_dict(json.loads(p.read_text())) for p in files]
        builder = DatasetBuilder(seed=self.train_cfg["training"]["seed"])
        splits = builder.split(builder.build_pairs(outputs))
        builder.save(splits, instructions)
        for name, records in splits.items():
            logger.info("  %-5s %d pairs", name, len(records))

    # -- stage 5: train --------------------------------------------------------

    def _run_train(self) -> None:
        from src.training.train import train

        logger.info("QLoRA fine-tuning %s (requires GPU)", self.train_cfg["model"]["name"])
        train(self.training_config_path, str(self.path("instructions")))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PD multi-omics training pipeline")
    parser.add_argument("--stage", default="all", choices=STAGES + ["all"])
    parser.add_argument("--acq-config", default="configs/acquisition.yaml")
    parser.add_argument("--pre-config", default="configs/preprocessing.yaml")
    parser.add_argument("--int-config", default="configs/integration.yaml")
    parser.add_argument("--train-config", default="configs/training.yaml")
    parser.add_argument("--data-root", default=None,
                        help="Rebase raw/processed/integrated/instructions under this directory")
    parser.add_argument("--max-subjects", type=int, default=None,
                        help="Cap the cohort size (overrides cohort.max_subjects)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    Pipeline(
        acquisition_config=args.acq_config,
        preprocessing_config=args.pre_config,
        integration_config=args.int_config,
        training_config=args.train_config,
        data_root=args.data_root,
        max_subjects=args.max_subjects,
    ).run_stage(args.stage)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
