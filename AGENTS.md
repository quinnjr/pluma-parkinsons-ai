# AGENTS.md -- pluma-parkinsons-ai

## Project Overview

Standalone Python package (not a PluMA plugin) for Parkinson's disease multi-omics biomarker discovery and LLM fine-tuning. Two-stage pipeline:

- **Stage 1 (ML):** Acquire multi-omics data (GEO, NHANES; PPMI/EPA clients exist) across 8 modalities (genomics, transcriptomics, epigenomics, proteomics, metabolomics, microbiome, environmental, clinical), simulate missing modalities with flagged synthetic data, preprocess per-modality, integrate via MOFA+ latent factors + an XGBoost classifier with **out-of-fold** TreeSHAP attributions, and produce per-subject `Stage1Output` JSON summaries carrying full provenance.
- **Stage 2 (LLM):** Convert Stage1Outputs into instruction-response pairs (biomarker discovery, clinical prediction, cross-modal synthesis) **grounded in a curated, PubMed-verified knowledge base**, then QLoRA fine-tune Gemma 4 on those pairs.

The grounding rule is absolute: every sentence in a generated response traces to (a) the pipeline's own numbers, (b) a cited knowledge-base claim, or (c) an explicit statement that no curated evidence exists. There is no fourth category; do not add prose that invents mechanisms for uncurated features.

## Architecture

```
src/
  pipeline.py           -- top-level orchestrator (CLI stages: acquire/preprocess/integrate/build_instructions/train)
                           stages hand off via disk under a --data-root; all stages are wired, none are stubs
  models.py             -- domain dataclasses: Subject, OmicsProfile, BiomarkerHit, Provenance, Stage1Output
  utils.py              -- load_config, ensure_dir, save_jsonl, load_jsonl

  acquisition/          -- data downloaders: PPMI, GEO (GEOparse, incl. probe->gene mapping),
                           environmental (EPA AQS, NHANES), synthetic.py (flagged simulated modalities)
  preprocessing/        -- per-modality normalization: genomics, transcriptomics, epigenomics,
                           proteomics, metabolomics, microbiome, environmental, clinical
  integration/
    mofa.py             -- MOFAIntegrator (mofapy2 wrapper)
    snf.py              -- SNFIntegrator (snfpy wrapper; experimental, not wired into the pipeline)
    ensemble.py         -- OmicsEnsemble: single XGBClassifier on concatenated features;
                           fit_evaluate() returns an EnsembleFit with out-of-fold proba/SHAP,
                           CV AUC, and an out_of_fold flag (falls back flagged on tiny cohorts)
    stage1_builder.py   -- EnsembleFit -> per-subject Stage1Output (modality inference, z-scores, provenance)

  knowledge/
    kb.py               -- KnowledgeBase: whole-token alias matching (TGBA1P3 != GBA1); lookup()
                           returns None for anything uncurated and callers must abstain, not improvise
    entities.yaml       -- 26 curated entities (genes, gut taxa, metabolites, exposures, scales)
    citations.yaml      -- 32 PubMed-verified citations; every pd_association must cite >=1
    verify_citations.py -- re-verify all PMIDs against PubMed (python -m src.knowledge.verify_citations)

  instruction_builder/
    templates.py        -- SYSTEM_PROMPT + per-task instruction variants
    formatter.py        -- InstructionFormatter: grounds hits against the KB, emits citations,
                           concordance verdicts, caveat blocks, and explicit abstentions
    dataset_builder.py  -- SUBJECT-level train/val/test split (never split one subject's pairs), JSONL

  training/
    prompts.py          -- message-list construction; chat format is rendered by the tokenizer's
                           apply_chat_template, never hand-written (Gemma 4 turn markers)
    model_utils.py      -- metrics: token_f1, citation hallucination audit, biomarker recall,
                           diagnosis extraction (no exact-match metric, deliberately)
    train.py            -- QLoRA: BitsAndBytes 4-bit NF4, LoRA adapters, TRL SFTTrainer
                           (prompt-completion records, loss on completion only, packing off)
    evaluate.py         -- generate on the held-out split and compute the metrics above

configs/                -- YAML configs for each stage
scripts/smoke_test.py   -- thin wrapper that runs the real Pipeline against data/smoke_test
                           (GSE6613 transcriptomics + NHANES blood lead, rest simulated)
tests/                  -- pytest suite mirroring src/ structure (incl. tests/knowledge/)
```

## Build and Dependencies

- **Build backend:** Hatchling (`pyproject.toml`).
- **Install:** `pip install -e ".[dev]"` for dev/test; `pip install -e ".[training]"` adds GPU deps.
- **Python:** >=3.11.
- **Core deps:** pandas, numpy, scikit-learn, xgboost, shap, mofapy2, snfpy, cyvcf2, GEOparse, pysradb, pydeseq2, scikit-bio, biopython, scipy, statsmodels, pyyaml, requests, tqdm.
- **Training deps (optional):** torch, transformers>=5.10 (gemma4 support), peft>=0.20, trl>=1.10, bitsandbytes>=0.48, datasets>=4, accelerate.
- **Dev deps:** pytest, pytest-cov, ruff.

## Conventions

- **Language:** Python 3.11+. Uses `from __future__ import annotations`, dataclasses, type hints.
- **Imports:** `from src.` prefixed (package-style). Heavy optional deps (torch, GEOparse, pandas in acquisition) are imported lazily inside functions.
- **Naming:** snake_case modules; PascalCase classes (`OmicsEnsemble`, `MOFAIntegrator`, `Stage1Output`).
- **Linter:** Ruff, line-length 100, rules `E`/`F`/`I`/`UP`, ignores `E501`.
- **Configs:** YAML files under `configs/` loaded via `src.utils.load_config`.
- **Data formats:** CSV for feature matrices, JSON for per-subject Stage1Output, JSONL for instruction pairs.
- **Label encoding:** `OmicsEnsemble.LABEL_MAP` maps PD -> 1, HC/SWEDD/Prodromal -> 0.
- **Chat format:** never hand-roll Gemma turn markers; always go through `apply_chat_template` / TRL prompt-completion records.
- **No MuPDF.** Use Micropdf if PDF handling is ever needed.

## Testing

- **Framework:** pytest (`testpaths = ["tests"]` in `pyproject.toml`).
- **Run tests:** `pip install -e ".[dev]" && pytest -v`
- **Coverage:** acquisition clients, preprocessing modules, MOFA alignment, ensemble (incl. out-of-fold behaviour and tiny-cohort fallback), stage1 builder (incl. modality inference and provenance), knowledge base (lookup/abstention and loader validation), instruction formatter (grounding, citations, caveats, no-fabrication), dataset builder (subject-level split), prompts, metrics, pipeline wiring, domain models.
- **Not covered:** networked smoke test (run `scripts/smoke_test.py` manually), `train.py` / `evaluate.py` model loading (requires GPU), SNF integration, full MOFA fit_transform.

## CLI

```bash
python -m src.pipeline --stage <stage>
```

Stages: `acquire`, `preprocess`, `integrate`, `build_instructions`, `train`, `all`.

Flags: `--acq-config`, `--pre-config`, `--int-config`, `--train-config`, `--data-root <dir>` (relocate all stage inputs/outputs), `--max-subjects N` (reduced run).

Direct training: `python -m src.training.train --config configs/training.yaml --data_dir data/instructions`

Evaluation: `python -m src.training.evaluate --adapter <path> --data data/instructions/test.jsonl` (or `--base-only`).

Citation drift check: `python -m src.knowledge.verify_citations` (exits 1 on drift).

## Configuration Files

- `configs/acquisition.yaml` -- PPMI base URL, GEO search terms/accessions, EPA/NHANES data dirs and cycles, synthetic-modality settings.
- `configs/preprocessing.yaml` -- per-modality parameters (MAF thresholds, normalization methods, known PD genes, rarefaction depth, UPDRS subscales, etc.).
- `configs/integration.yaml` -- MOFA (n_factors, max_iter), SNF (experimental, unused), ensemble (n_estimators, max_depth, learning_rate, cv_folds, top_k_biomarkers).
- `configs/training.yaml` -- model name (`google/gemma-4-12B-it`; 31B / 26B-A4B / E4B alternatives listed), BitsAndBytes 4-bit NF4 config, LoRA params (r=32, alpha=64, suffix-named projection targets so vision/audio towers stay frozen), training hyperparams (3 epochs, batch 2, grad accum 8, lr 2e-4, cosine schedule, max_length 4096, packing off).

## Domain Models (src/models.py)

- `Subject` -- subject_id, diagnosis (PD/HC/SWEDD/Prodromal), age, sex.
- `OmicsProfile` -- modality name + feature dict.
- `BiomarkerHit` -- modality, feature, shap_value, `effect` (`toward_pd`/`away_from_pd`: which way the feature pushed the *classifier*), `value_z` (the subject's standardised measurement, or None). The SHAP sign and the measurement direction are deliberately separate fields; `from_dict` still reads the legacy `direction` key.
- `Provenance` -- cohort_size, shap_out_of_fold, synthetic_modalities, datasets, model, cv_auc. Attached to every `Stage1Output` so generated text can be honest about its evidence.
- `Stage1Output` -- subject_id, diagnosis, prediction_confidence, disease_stage (early/mid/late/None; None when the source data has no staging — never fabricate), top_biomarkers, mofa_factors, environmental_risk_score, provenance; serializable to/from JSON.

## Instruction Templates

Three task types generated per subject (`src/instruction_builder/`):
1. **Biomarker discovery** -- ranked attributions with per-feature KB annotation or explicit abstention, concordance vs the literature, caveats, references.
2. **Clinical prediction** -- model output framed as classifier output (not a diagnosis), supporting/opposing features, evidential-weight statement.
3. **Cross-modal synthesis** -- convergent/conflicting signals across modalities; abstains entirely when no annotated features exist.

Each uses seeded randomized instruction variants from `templates.py`. Prompts are message lists (system/user/assistant) rendered through the tokenizer chat template — **not** Alpaca `### Instruction:` markers.

## Smoke Test

`scripts/smoke_test.py [--subjects 20]` -- runs the real `Pipeline` (all stages except `train`) against `data/smoke_test/`: GSE6613 whole-blood transcriptomics (real GEO download), NHANES blood lead (real XPT), everything else simulated and flagged. Prints grounding statistics from the produced artefacts. Requires network.

## Known Issues and Discrepancies

- **SNF integration:** `src/integration/snf.py` exists and has a config block, but is not exercised by the pipeline or smoke test and has no tests. Treat it as experimental.
- **PPMI/EPA acquisition:** clients exist but the default pipeline path only downloads GEO + NHANES; PPMI requires registered credentials (see README).
- **Transductive feature construction:** MOFA+ factors, top-variable-gene selection, and the cohort z-scores are fit on the full cohort before the ensemble's CV split. The leak is unsupervised (no labels) but means `out_of_fold=True` guarantees honest *label* handling only, not fully out-of-sample features.
- **Synthetic-modality label dependence:** `synthetic.py` shifts curated features by diagnosis, so any run with simulated modalities has an inflated CV AUC by construction. The generated text now states this (profile AUC qualifier + always-on caveat); the number itself is still not a performance estimate.
- **Citation audit scope:** `audit_citations` checks that a cited PMID exists in the KB, not that it is attached to the right entity or claim. Entity-level attachment checking is future work; read `citation_hallucination_rate_micro` together with `responses_with_citations`.
- **Environmental risk score:** NHANES values are assigned to cohort subjects by position (a distribution, not a measurement); the pipeline flags `environmental` as simulated and the prompt line carries a qualifier, but the score is still a single unqualified float in `Stage1Output`.
- **`train.py` GPU path untested:** requires `[training]` extras and a GPU. Training precision now derives from `bnb_4bit_compute_dtype` (set it to `float16` for pre-Ampere cards). The TRL 1.10 kwargs (`SFTTrainer(quantization_config=...)`, `SFTConfig(max_length/packing/completion_only_loss)`, `train_sampling_strategy`) were verified against TRL/transformers source, not by execution.

## No Relationship to PluMA Plugin Runtime

Despite the "PluMA" branding, this project is a standalone Python package. It does not import or orchestrate any PluMA plugins, and does not follow the PluMA `input()` / `run()` / `output()` plugin contract.

## Attribution

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
License: MIT
