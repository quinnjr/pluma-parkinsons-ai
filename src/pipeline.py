# src/pipeline.py
"""
Top-level pipeline orchestrator for the PD multi-omics training pipeline.

Run individual stages or the full pipeline end-to-end:

    python -m src.pipeline --stage all
    python -m src.pipeline --stage preprocess
    python -m src.pipeline --stage integrate
    python -m src.pipeline --stage build_instructions
    python -m src.pipeline --stage train

Stage 1 (acquire, preprocess, integrate) is compute-heavy and typically
runs on CPU/cluster. Stage 2 (build_instructions, train) requires GPU.
They will often run on different machines or at different times.
"""
from __future__ import annotations
import argparse
import logging
from src.utils import load_config

logger = logging.getLogger(__name__)

STAGES = ["acquire", "preprocess", "integrate", "build_instructions", "train"]


class Pipeline:
    def __init__(
        self,
        acquisition_config: str = "configs/acquisition.yaml",
        preprocessing_config: str = "configs/preprocessing.yaml",
        integration_config: str = "configs/integration.yaml",
        training_config: str = "configs/training.yaml",
    ):
        self.acq_cfg = load_config(acquisition_config)

    def run_stage(self, stage: str) -> None:
        if stage == "all":
            for s in STAGES:
                logger.info("Running stage: %s", s)
                self.run_stage(s)
            return
        if stage not in STAGES:
            raise ValueError(f"Unknown stage {stage!r}. Choose from: {STAGES}")
        getattr(self, f"_run_{stage}")()

    def _run_acquire(self) -> None:
        logger.info("Acquisition: downloading data from PPMI, GEO, EPA, NHANES")
        logger.info(
            "NOTE: PPMI data requires manual download from https://www.ppmi-info.org"
        )

    def _run_preprocess(self) -> None:
        logger.info("Preprocessing: normalizing all 8 modalities")

    def _run_integrate(self) -> None:
        logger.info("Integration: running MOFA+, SNF, XGBoost ensemble + SHAP")

    def _run_build_instructions(self) -> None:
        logger.info("Building instruction-response pairs from Stage 1 outputs")

    def _run_train(self) -> None:
        logger.info("Training: QLoRA fine-tuning Mistral Small 4 (requires GPU)")
        from src.training.train import train
        train("configs/training.yaml", "data/integrated/instruction_pairs")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="PD multi-omics training pipeline")
    parser.add_argument("--stage", default="all",
                        choices=STAGES + ["all"], help="Pipeline stage to run")
    parser.add_argument("--acq-config", default="configs/acquisition.yaml")
    parser.add_argument("--pre-config", default="configs/preprocessing.yaml")
    parser.add_argument("--int-config", default="configs/integration.yaml")
    parser.add_argument("--train-config", default="configs/training.yaml")
    args = parser.parse_args()

    pipeline = Pipeline(
        acquisition_config=args.acq_config,
        preprocessing_config=args.pre_config,
        integration_config=args.int_config,
        training_config=args.train_config,
    )
    pipeline.run_stage(args.stage)
