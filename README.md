# PluMA Parkinson's AI — Multi-Omics Training Pipeline

Fine-tunes **Gemma 4** for Parkinson's disease **biomarker discovery** and **clinical prediction** using integrated multi-omics data from public databases, with every generated claim grounded in a curated, PubMed-verified knowledge base.

## Architecture

**Stage 1 (ML layer):** Downloads and preprocesses omics modalities → MOFA+ latent factors → XGBoost classifier with **out-of-fold** TreeSHAP attributions → per-subject `Stage1Output` JSON summaries carrying full provenance (cohort size, in-sample vs out-of-fold SHAP, simulated modalities, source datasets, CV AUC).

**Stage 2 (LLM layer):** Converts `Stage1Output`s to instruction-response pairs grounded in `src/knowledge/` (26 curated entities, 32 PubMed-verified citations) → QLoRA fine-tuning of `google/gemma-4-12B-it`.

Every sentence in a generated response traces to one of exactly three sources: the pipeline's own numbers, a cited knowledge-base claim, or an explicit statement that no curated evidence exists. Features the knowledge base cannot resolve are reported as statistical associations only — never given an invented mechanism.

## Setup

```bash
# Core + dev dependencies
pip install -e ".[dev]"

# Add LLM training dependencies (requires GPU)
pip install -e ".[training]"
```

## Data Sources

| Modality | Sources |
|---|---|
| Genomics | PPMI WGS, GP2, GWAS Catalog, dbGaP, UK Biobank |
| Transcriptomics | PPMI RNA-seq, GEO, Allen Brain Atlas |
| Epigenomics | GEO methylation arrays, ENCODE, Roadmap Epigenomics |
| Proteomics | PPMI CSF proteomics, Human Protein Atlas, PRIDE Archive |
| Metabolomics | PPMI metabolomics, MetaboLights, HMDB |
| Microbiome | PPMI gut microbiome, EBI Metagenomics, GEO 16S |
| Environmental | EPA AQS, USGS pesticide maps, CDC NHANES, NIH NTP |
| Clinical/Phenotypic | PPMI clinical, OpenNeuro, PhysioNet |

Modalities without a real data source in a given run can be simulated (`src/acquisition/synthetic.py`); simulated modalities are recorded in each subject's provenance and called out in the generated text.

### Data Access Requirements

Some sources require registration before data can be downloaded. The table below lists what is needed and where to sign up.

#### Requires registration

| Source | Cost | Approval time | Instructions |
|--------|------|---------------|--------------|
| **PPMI** | Free | 1–3 business days | Create an account at [ppmi-info.org](https://www.ppmi-info.org/access-data-specimens/download-data), agree to the Data Use Agreement, then download Demographics, genomics, proteomics, metabolomics, microbiome, and clinical CSVs into `data/raw/ppmi/`. |
| **EPA AQS API** | Free | Same day (email) | Submit a key request at [aqs.epa.gov](https://aqs.epa.gov/aqsweb/documents/data_api.html#signup). Set the returned key as `EPA_AQS_KEY` in `.env`. |
| **GP2** | Free | Variable | Register at [gp2.org](https://gp2.org) and request access to the genotyping dataset. |
| **dbGaP** | Free | Weeks–months (NIH review) | Create an [eRA Commons account](https://public.era.nih.gov/commons/), then submit a controlled-access request for each study of interest at [dbgap.ncbi.nlm.nih.gov](https://dbgap.ncbi.nlm.nih.gov). |
| **UK Biobank** | Fee per project | Weeks–months | Apply at [ukbiobank.ac.uk](https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access). |
| **PhysioNet** (some datasets) | Free | Days | Register at [physionet.org](https://physionet.org) and complete credentialing for datasets that require it. |

#### No credentials needed

GEO, NHANES, GWAS Catalog, ENCODE, Roadmap Epigenomics, Human Protein Atlas, PRIDE Archive, MetaboLights, HMDB, EBI Metagenomics (MGnify), OpenNeuro, Allen Brain Atlas, and USGS pesticide maps are all downloaded automatically with no sign-up.

#### Priority

For a first training run, **PPMI** is the most important — it provides longitudinal multi-omics from the same subjects across all modalities. The **EPA AQS key** is quick to obtain and improves environmental exposure coverage. Everything else is optional for an initial run.

## Running the Pipeline

```bash
# Stage 1 — data acquisition and preprocessing (CPU)
python -m src.pipeline --stage acquire
python -m src.pipeline --stage preprocess
python -m src.pipeline --stage integrate

# Build LLM training data (subject-level train/val/test split)
python -m src.pipeline --stage build_instructions

# Stage 2 — QLoRA fine-tuning (4-bit NF4; gemma-4-12B-it fits a 24 GB card)
python -m src.pipeline --stage train

# Full pipeline end-to-end
python -m src.pipeline --stage all

# Useful flags: --data-root <dir> to relocate all inputs/outputs,
# --max-subjects N for a reduced run, --acq/pre/int/train-config overrides.
```

Stages hand off via disk (`<data-root>/{raw,processed,integrated,instructions}`), so each stage can be run and inspected independently.

## Evaluation

```bash
python -m src.training.evaluate --adapter models/gemma-4-12b-pd-multiomics \
    --data data/instructions/test.jsonl --limit 50
# add --base-only to score the un-tuned base model as a baseline
```

Reported metrics: token F1 against the reference, **citation hallucination rate** (PMIDs cited that the knowledge base does not contain), **biomarker recall** (pipeline-ranked features actually named), and diagnosis accuracy on the prediction task. Exact-match accuracy is deliberately not reported.

To re-verify every knowledge-base citation against PubMed:

```bash
python -m src.knowledge.verify_citations   # exits 1 on drift
```

## Tests

```bash
.venv/bin/pytest -v
```

## Project Structure

```
pluma-parkinsons-ai/
├── src/
│   ├── acquisition/          # Database downloaders (PPMI, GEO, EPA, NHANES) + synthetic modalities
│   ├── preprocessing/        # Per-modality normalizers (8 modalities)
│   ├── integration/          # MOFA+, XGBoost + out-of-fold SHAP, Stage1Builder
│   ├── knowledge/            # Curated entities + PubMed-verified citations (grounding)
│   ├── instruction_builder/  # Stage1Output → grounded instruction pairs, JSONL splits
│   └── training/             # QLoRA fine-tuning (Gemma 4), chat prompts, evaluation
├── configs/                  # YAML configs for each pipeline stage
├── data/                     # Raw → processed → integrated → instructions (gitignored)
├── scripts/                  # smoke_test.py — end-to-end run on a small real cohort
└── tests/                    # Mirror of src/ structure
```
