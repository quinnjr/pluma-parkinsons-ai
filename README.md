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
