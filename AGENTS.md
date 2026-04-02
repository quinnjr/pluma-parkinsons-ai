# AGENTS.md -- pluma-parkinsons-ai

## Project Overview

Standalone Python package (not a PluMA plugin) for Parkinson's disease multi-omics biomarker discovery and LLM fine-tuning. Two-stage pipeline:

- **Stage 1 (ML):** Acquire multi-omics data (PPMI, GEO, EPA, NHANES) across 8 modalities (genomics, transcriptomics, epigenomics, proteomics, metabolomics, microbiome, environmental, clinical), preprocess per-modality, integrate via MOFA+ latent factors + XGBoost ensemble with SHAP, produce per-subject `Stage1Output` JSON summaries.
- **Stage 2 (LLM):** Convert Stage1Outputs into instruction-response pairs (biomarker discovery, clinical prediction, cross-modal synthesis), QLoRA fine-tune a Mistral-class instruct model on those pairs.

## Architecture

```
src/
  pipeline.py           -- top-level orchestrator (CLI stages: acquire/preprocess/integrate/build_instructions/train)
  models.py             -- domain dataclasses: Subject, OmicsProfile, BiomarkerHit, Stage1Output
  utils.py              -- load_config, ensure_dir, save_jsonl, load_jsonl

  acquisition/          -- data downloaders: PPMI, GEO (GEOparse), environmental (EPA AQS, NHANES)
  preprocessing/        -- per-modality normalization: genomics, transcriptomics, epigenomics,
                           proteomics, metabolomics, microbiome, environmental, clinical
  integration/
    mofa.py             -- MOFAIntegrator (mofapy2 wrapper)
    snf.py              -- SNFIntegrator (snfpy wrapper)
    ensemble.py         -- OmicsEnsemble (XGBClassifier + SHAP on concatenated features)
    stage1_builder.py   -- assembles per-subject Stage1Output JSON from integration results

  instruction_builder/
    templates.py        -- instruction prompt templates (biomarker, prediction, synthesis)
    formatter.py        -- InstructionFormatter: Stage1Output -> instruction-response dicts
    dataset_builder.py  -- splits and writes train/val/test JSONL

  training/
    model_utils.py      -- format_prompt (Alpaca-style), compute_metrics
    train.py            -- QLoRA fine-tuning: BitsAndBytes 4-bit, LoRA adapters, SFTTrainer

configs/                -- YAML configs for each stage
scripts/smoke_test.py   -- end-to-end smoke test (GEO + NHANES + synthetic modalities)
tests/                  -- pytest suite mirroring src/ structure
data/smoke_test/        -- committed sample data (20 integrated JSONs, train/val/test JSONL)
```

## Build and Dependencies

- **Build backend:** Hatchling (`pyproject.toml`).
- **Install:** `pip install -e ".[dev]"` for dev/test; `pip install -e ".[training]"` adds GPU deps.
- **Python:** >=3.11.
- **Core deps:** pandas, numpy, scikit-learn, xgboost, shap, mofapy2, snfpy, cyvcf2, GEOparse, pysradb, pydeseq2, scikit-bio, biopython, scipy, statsmodels, pyyaml, requests, tqdm.
- **Training deps (optional):** torch, transformers, peft, trl, bitsandbytes, datasets, accelerate, evaluate.
- **Dev deps:** pytest, pytest-cov, ruff.

## Conventions

- **Language:** Python 3.11+. Uses `from __future__ import annotations`, dataclasses, type hints.
- **Imports:** `from src.` prefixed (package-style).
- **Naming:** snake_case modules; PascalCase classes (`OmicsEnsemble`, `MOFAIntegrator`, `Stage1Output`).
- **Linter:** Ruff, line-length 100, rules `E`/`F`/`I`/`UP`, ignores `E501`.
- **Configs:** YAML files under `configs/` loaded via `src.utils.load_config`.
- **Data formats:** CSV for feature matrices, JSON for per-subject Stage1Output, JSONL for instruction pairs.
- **Label encoding:** `OmicsEnsemble.LABEL_MAP` maps PD -> 1, HC/SWEDD/Prodromal -> 0.
- **No MuPDF.** Use Micropdf if PDF handling is ever needed.

## Testing

- **Framework:** pytest (`testpaths = ["tests"]` in `pyproject.toml`).
- **Run tests:** `pip install -e ".[dev]" && pytest -v`
- **Coverage:** Tests cover acquisition clients, preprocessing modules, MOFA alignment, ensemble, stage1 builder, instruction formatter, dataset builder, model utils, pipeline wiring, domain models.
- **Not covered:** End-to-end smoke test (requires network), `train.py` (requires GPU), SNF integration, full MOFA fit_transform.

## CLI

```bash
python -m src.pipeline --stage <stage>
```

Stages: `acquire`, `preprocess`, `integrate`, `build_instructions`, `train`, `all`.

Config overrides: `--acq-config`, `--pre-config`, `--int-config`, `--train-config`.

Direct training: `python -m src.training.train --config configs/training.yaml --data_dir data/integrated/instruction_pairs`

## Configuration Files

- `configs/acquisition.yaml` -- PPMI base URL, GEO search terms, EPA/NHANES data dirs and cycles.
- `configs/preprocessing.yaml` -- per-modality parameters (MAF thresholds, normalization methods, known PD genes, rarefaction depth, UPDRS subscales, etc.).
- `configs/integration.yaml` -- MOFA (n_factors, max_iter), SNF (k_neighbors, mu, iterations), ensemble (base models per modality, meta-model, CV folds, top features).
- `configs/training.yaml` -- model name (`mistralai/Mistral-Small-3.1-24B-Instruct-2503`), BitsAndBytes 4-bit NF4 config, LoRA params (r=32, alpha=64, target modules), training hyperparams (3 epochs, batch 2, grad accum 8, lr 2e-4, cosine schedule, max_seq_length 4096).

## Domain Models (src/models.py)

- `Subject` -- subject_id, diagnosis (PD/HC/SWEDD/Prodromal), age, sex.
- `OmicsProfile` -- modality name + feature dict.
- `BiomarkerHit` -- modality, feature, shap_value, direction; serializable to/from dict.
- `Stage1Output` -- subject_id, diagnosis, prediction_confidence, disease_stage (early/mid/late/None), top_biomarkers, mofa_factors, environmental_risk_score; serializable to/from JSON.

## Instruction Templates

Three task types generated per subject (`src/instruction_builder/`):
1. **Biomarker discovery** -- identify and explain top biomarkers from the multi-omics profile.
2. **Clinical prediction** -- predict PD diagnosis and stage with reasoning.
3. **Cross-modal synthesis** -- describe convergent signals across modalities.

Each uses randomized instruction variants from `templates.py`. Prompt format is Alpaca-style (`### Instruction / ### Input / ### Response`).

## Smoke Test

`scripts/smoke_test.py` -- fetches real GEO data (GSE153404 / GSE153405), NHANES environmental data, generates synthetic modalities for missing data, runs MOFA + ensemble + SHAP + stage1 builder + instruction builder. Produces `data/smoke_test/integrated/*.json` and `data/smoke_test/instructions/*.jsonl`.

## Known Issues and Discrepancies

- **Pipeline stubs:** `_run_acquire`, `_run_preprocess`, `_run_integrate`, `_run_build_instructions` in `pipeline.py` only log messages. They do not call the actual acquisition/preprocessing/integration/instruction modules. Only `_run_train` is wired. Running `--stage all` does not reproduce the smoke test.
- **Pipeline config loading:** `Pipeline.__init__` accepts four config paths but only loads `acquisition_config` into `self.acq_cfg`; preprocessing/integration/training configs are not stored or used.
- **Model naming:** README and `train.py` docstring say "Mistral Small 4"; `configs/training.yaml` specifies `Mistral-Small-3.1-24B-Instruct-2503`.
- **`cross_modal_synthesis`:** Claims two features "both contributing in the same direction" regardless of their actual SHAP sign directions.
- **`biomarker_discovery`:** Uses "known contributor to PD pathophysiology" for all features including HC (healthy control) profiles.
- **`prepare_dataset` in train.py:** Accepts `max_seq_length` parameter but does not use it in the dataset map.
- **`notebooks/` directory:** Referenced in README but does not exist.
- **SNF integration:** `src/integration/snf.py` exists and is configured in `integration.yaml`, but is not exercised by the smoke test or pipeline, and has no tests.
- **`configs/integration.yaml` ensemble:** Describes per-modality base models and a meta-model, but `OmicsEnsemble` is a single `XGBClassifier` on concatenated features (not a stacking ensemble).

## No Relationship to PluMA Plugin Runtime

Despite the "PluMA" branding, this project is a standalone Python package. It does not import or orchestrate any PluMA plugins, and does not follow the PluMA `input()` / `run()` / `output()` plugin contract.

## Attribution

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
License: MIT
