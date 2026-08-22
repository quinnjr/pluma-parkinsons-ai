"""End-to-end smoke test: run the real pipeline against a small public cohort.

This runs `src.pipeline` itself rather than reimplementing it. If the pipeline
breaks, this breaks, which is the only arrangement worth having — the previous
version duplicated the whole pipeline body, so it could pass while
`python -m src.pipeline --stage all` was still four logging stubs.

Data used:
  - Transcriptomics: GSE6613 (GEO), whole-blood expression, real download
  - Environmental:   NHANES blood lead (real public XPT)
  - Everything else: simulated, flagged as such in every artefact produced

Run:
    .venv/bin/python scripts/smoke_test.py [--subjects 20]

Outputs land under data/smoke_test/ and leave the main data directories alone.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import Pipeline  # noqa: E402  (after sys.path setup)
from src.utils import load_jsonl  # noqa: E402

log = logging.getLogger("smoke_test")

DATA_ROOT = ROOT / "data" / "smoke_test"
STAGES = ["acquire", "preprocess", "integrate", "build_instructions"]


def summarise(data_root: Path) -> None:
    """Print what the run produced, including the grounding statistics."""
    integrated = sorted((data_root / "integrated").glob("*.json"))
    log.info("Stage1Output records: %d", len(integrated))
    if integrated:
        sample = json.loads(integrated[0].read_text())
        provenance = sample["provenance"]
        log.info("  subject               %s (%s)", sample["subject_id"], sample["diagnosis"])
        log.info("  predicted P(PD)       %.3f", sample["prediction_confidence"])
        log.info("  SHAP out-of-fold      %s", provenance["shap_out_of_fold"])
        log.info("  cross-validated AUC   %s", provenance["cv_auc"])
        log.info("  simulated modalities  %s", ", ".join(provenance["synthetic_modalities"]))
        for hit in sample["top_biomarkers"][:3]:
            log.info(
                "    [%s] %s SHAP=%+.3f %s",
                hit["modality"], hit["feature"], hit["shap"], hit["effect"],
            )

    train = load_jsonl(data_root / "instructions" / "train.jsonl")
    val = load_jsonl(data_root / "instructions" / "val.jsonl")
    test = load_jsonl(data_root / "instructions" / "test.jsonl")
    log.info("Instruction pairs: %d train / %d val / %d test", len(train), len(val), len(test))

    subjects = {split: {r["subject_id"] for r in recs}
                for split, recs in (("train", train), ("val", val), ("test", test))}
    overlap = (
        (subjects["train"] & subjects["val"])
        | (subjects["train"] & subjects["test"])
        | (subjects["val"] & subjects["test"])
    )
    if overlap:
        raise AssertionError(f"subject leakage across splits: {sorted(overlap)}")
    log.info("No subject appears in more than one split.")

    grounded = [r for r in train if r["grounding"]["pmids"]]
    log.info(
        "Grounding: %d/%d training pairs cite at least one verified PMID; "
        "%d distinct PMIDs used",
        len(grounded), len(train),
        len({p for r in train for p in r["grounding"]["pmids"]}),
    )
    # The simulated modalities carry curated, KB-annotated features by design,
    # so a run in which NO pair cites a PMID means grounding is broken —
    # exactly the regression this smoke test exists to catch.
    if not grounded:
        raise AssertionError(
            "no training pair cites a verified PMID; knowledge-base grounding "
            "is not reaching the generated pairs"
        )
    if train:
        sample = train[0]
        log.info("\n--- sample pair (%s) ---", sample["task"])
        log.info("INSTRUCTION: %s", sample["instruction"])
        log.info("INPUT (head):\n%s", "\n".join(sample["input"].splitlines()[:8]))
        log.info("OUTPUT (head):\n%s", "\n".join(sample["output"].splitlines()[:12]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, default=20, help="cohort size cap")
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("=" * 70)
    log.info("PluMA Parkinson's AI — end-to-end smoke test (real GEO + NHANES)")
    log.info("=" * 70)

    pipeline = Pipeline(
        acquisition_config=str(ROOT / "configs" / "acquisition.yaml"),
        preprocessing_config=str(ROOT / "configs" / "preprocessing.yaml"),
        integration_config=str(ROOT / "configs" / "integration.yaml"),
        training_config=str(ROOT / "configs" / "training.yaml"),
        data_root=args.data_root,
        max_subjects=args.subjects,
    )
    for stage in STAGES:
        log.info("--- stage: %s ---", stage)
        pipeline.run_stage(stage)

    summarise(Path(args.data_root))
    log.info("=" * 70)
    log.info("Smoke test PASSED. Stage 2 training is not run here (needs a GPU).")
    log.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
