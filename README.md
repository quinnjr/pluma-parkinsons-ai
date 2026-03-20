# PluMA Parkinson's AI — Multi-Omics Training Pipeline

Fine-tunes Mistral Small 4 for Parkinson's disease **biomarker discovery** and **clinical prediction** using integrated multi-omics data from public databases.

## Architecture

**Stage 1 (ML layer):** Downloads and preprocesses 8 omics modalities → MOFA+/SNF integration → XGBoost ensemble + SHAP → per-sample JSON summaries

**Stage 2 (LLM layer):** Converts JSON summaries to instruction-response pairs → QLoRA fine-tuning of Mistral Small 4

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

### PPMI Registration (Required)

PPMI data requires free registration at **https://www.ppmi-info.org/access-data-specimens/download-data**

After registering, download Demographics, genomics, proteomics, metabolomics, microbiome, and clinical CSVs and place them in `data/raw/ppmi/`.

All other data sources are downloaded automatically.

## Running the Pipeline

```bash
# Stage 1 — data acquisition and preprocessing (CPU)
python -m src.pipeline --stage acquire
python -m src.pipeline --stage preprocess
python -m src.pipeline --stage integrate

# Build LLM training data
python -m src.pipeline --stage build_instructions

# Stage 2 — QLoRA fine-tuning (requires ≥20GB VRAM)
python -m src.pipeline --stage train

# Full pipeline end-to-end
python -m src.pipeline --stage all
```

## Tests

```bash
.venv/bin/pytest -v
```

## Project Structure

```
pluma-parkinsons-ai/
├── src/
│   ├── acquisition/          # Database downloaders (PPMI, GEO, EPA, NHANES)
│   ├── preprocessing/        # Per-modality normalizers (8 modalities)
│   ├── integration/          # MOFA+, SNF, XGBoost ensemble, Stage1Builder
│   ├── instruction_builder/  # Stage1Output → instruction pairs, JSONL splits
│   └── training/             # QLoRA training script for Mistral Small 4
├── configs/                  # YAML configs for each pipeline stage
├── data/                     # Raw → processed → integrated (gitignored)
├── notebooks/                # EDA and validation notebooks
└── tests/                    # Mirror of src/ structure
```
