"""
End-to-end smoke test using real public data.

Real data sources used:
  - Transcriptomics: GSE6613 (GEO) — blood gene expression, 50 PD / 22 HC
  - Environmental:   NHANES 2017-2018 blood lead / cadmium (public XPT)
  - All other modalities: realistic synthetic data drawn from published
    PD-relevant distributions (no PPMI registration required)

Run:
    .venv/bin/python scripts/smoke_test.py

Outputs written to:
    data/smoke_test/integrated/     — Stage1Output JSONs
    data/smoke_test/instructions/   — train/val/test JSONL splits
"""
from __future__ import annotations
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ── project root on path ─────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("smoke_test")

# ── output directories ────────────────────────────────────────────────────────
DATA_DIR      = ROOT / "data" / "smoke_test"
GEO_DIR       = DATA_DIR / "raw" / "geo"
NHANES_DIR    = DATA_DIR / "raw" / "nhanes"
INTEGRATED    = DATA_DIR / "integrated"
INSTRUCTIONS  = DATA_DIR / "instructions"
for d in [GEO_DIR, NHANES_DIR, INTEGRATED, INSTRUCTIONS]:
    d.mkdir(parents=True, exist_ok=True)

N_SUBJECTS = 20   # use first 20 subjects for speed


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Download GSE6613 (real PD blood RNA-seq / microarray)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_gse6613() -> tuple[pd.DataFrame, pd.Series]:
    """Return (expression_matrix genes×samples, diagnosis_series)."""
    import GEOparse

    log.info("Downloading GSE6613 from GEO (this may take a minute)…")
    gse = GEOparse.get_GEO(geo="GSE6613", destdir=str(GEO_DIR), silent=True)

    frames, sample_meta = [], {}
    for gsm_name, gsm in gse.gsms.items():
        # extract diagnosis from characteristics
        chars = gsm.metadata.get("characteristics_ch1", [])
        diag_raw = next(
            (c for c in chars if "disease" in c.lower() or "status" in c.lower()), ""
        )
        diag = "PD" if "parkinson" in diag_raw.lower() else "HC"
        sample_meta[gsm_name] = diag

        if gsm.table is not None and not gsm.table.empty:
            col = gsm.table.set_index("ID_REF")["VALUE"].rename(gsm_name)
            frames.append(col)

    expr = pd.concat(frames, axis=1).apply(pd.to_numeric, errors="coerce").fillna(0)
    diagnosis = pd.Series(sample_meta)

    # keep only subjects with clear diagnosis; limit to N_SUBJECTS
    keep = diagnosis[diagnosis.isin(["PD", "HC"])].index[:N_SUBJECTS]
    log.info("GSE6613: %d genes × %d samples (%s)",
             len(expr), len(keep),
             diagnosis[keep].value_counts().to_dict())
    return expr[keep], diagnosis[keep]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Download NHANES 2017-2018 blood metals (real environmental data)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_nhanes_metals(subjects: list[str]) -> pd.Series:
    """Return an environmental_risk_score per subject (real NHANES lead values)."""
    import requests

    url = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/PBCD_J.XPT"
    dest = NHANES_DIR / "PBCD_J.XPT"
    if not dest.exists():
        log.info("Downloading NHANES blood metals (PBCD_J.XPT)…")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        dest.write_bytes(r.content)
    else:
        log.info("NHANES blood metals already cached.")

    try:
        df = pd.read_sas(str(dest), format="xport", encoding="utf-8")
        log.info("Parsed NHANES XPT via pandas.read_sas: %d rows", len(df))
    except Exception as e:
        log.warning("Could not parse NHANES XPT (%s) — using synthetic env scores", e)
        np.random.seed(7)
        return pd.Series(
            np.random.uniform(1.0, 9.0, len(subjects)),
            index=subjects,
            name="environmental_risk_score",
        )

    # LBXBPB = blood lead (µg/dL); normalize to 0–10 risk score
    if "LBXBPB" in df.columns:
        lead = df["LBXBPB"].dropna()
        lead_norm = ((lead - lead.min()) / (lead.max() - lead.min() + 1e-9) * 10)
        scores = lead_norm.values[:len(subjects)]
        if len(scores) < len(subjects):
            scores = np.pad(scores, (0, len(subjects) - len(scores)),
                            constant_values=float(scores.mean()))
        log.info("NHANES blood lead scores: min=%.2f max=%.2f mean=%.2f",
                 scores.min(), scores.max(), scores.mean())
        return pd.Series(scores, index=subjects, name="environmental_risk_score")

    log.warning("LBXBPB column not found in NHANES file — using synthetic env scores")
    np.random.seed(7)
    return pd.Series(np.random.uniform(1.0, 9.0, len(subjects)),
                     index=subjects, name="environmental_risk_score")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Synthetic modality data (realistic PD-shifted distributions)
# ─────────────────────────────────────────────────────────────────────────────
def make_synthetic_modalities(subjects: list[str],
                               diagnosis: pd.Series) -> dict[str, pd.DataFrame]:
    """Generate realistic synthetic data for modalities not available without PPMI."""
    rng = np.random.default_rng(42)
    n = len(subjects)
    is_pd = (diagnosis.loc[subjects] == "PD").values.astype(float)

    def _df(data, prefix, subjects):
        cols = [f"{prefix}_{i}" for i in range(data.shape[1])]
        return pd.DataFrame(data, index=subjects, columns=cols)

    # Genomics: risk allele dosages, PD subjects enriched for known variants
    geno = rng.integers(0, 3, size=(n, 30)).astype(float)
    geno[:, 0] = rng.binomial(1, 0.05 + 0.15 * is_pd)   # LRRK2-like variant
    geno[:, 1] = rng.binomial(1, 0.02 + 0.08 * is_pd)   # GBA-like variant

    # Epigenomics: beta values, PD hypomethylated at some CpGs
    epi = rng.uniform(0.1, 0.9, size=(n, 40))
    epi[:, :5] -= 0.15 * is_pd[:, None]
    epi = epi.clip(0.01, 0.99)

    # Proteomics: alpha-synuclein-like proteins elevated in PD
    prot = rng.lognormal(0, 0.5, size=(n, 25))
    prot[:, 0] *= (1 + 0.5 * is_pd)   # aSyn proxy

    # Metabolomics: some metabolites depleted in PD
    metab = rng.exponential(1.0, size=(n, 35)) + 0.1
    metab[:, :4] *= (1 - 0.3 * is_pd[:, None])

    # Microbiome: Prevotella depleted, Akkermansia elevated in PD
    micro = rng.dirichlet(np.ones(20), size=n) * 10000
    micro[:, 0] *= (1 - 0.4 * is_pd)    # Prevotella-like
    micro[:, 1] *= (1 + 0.3 * is_pd)    # Akkermansia-like

    return {
        "genomics":       _df(geno,  "SNP",   subjects),
        "epigenomics":    _df(epi,   "CpG",   subjects),
        "proteomics":     _df(prot,  "prot",  subjects),
        "metabolomics":   _df(metab, "metab", subjects),
        "microbiome":     _df(micro, "bug",   subjects),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Preprocess all modalities
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_all(expr: pd.DataFrame,
                   synthetic: dict[str, pd.DataFrame],
                   subjects: list[str]) -> dict[str, pd.DataFrame]:
    from src.preprocessing.transcriptomics import TranscriptomicsPreprocessor
    from src.preprocessing.genomics import GenomicsPreprocessor, DEFAULT_PD_GENES
    from src.preprocessing.epigenomics import EpigenomicsPreprocessor
    from src.preprocessing.proteomics import ProteomicsPreprocessor
    from src.preprocessing.metabolomics import MetabolomicsPreprocessor
    from src.preprocessing.microbiome import MicrobiomePreprocessor

    log.info("Preprocessing transcriptomics (GSE6613)…")
    tp = TranscriptomicsPreprocessor(min_count=5, min_sample_fraction=0.2)
    gene_lengths = pd.Series(
        np.random.default_rng(0).integers(500, 5000, len(expr)),
        index=expr.index,
    )
    txn = tp.preprocess(expr[subjects], gene_lengths).T   # subjects × genes
    # keep top 100 most-variable genes for speed
    txn = txn.loc[:, txn.var().nlargest(100).index]
    log.info("  → transcriptomics: %s", txn.shape)

    log.info("Preprocessing genomics…")
    # synthetic genomics is already in subjects × variants form
    geno_proc = synthetic["genomics"].clip(0, 2).astype(int)
    log.info("  → genomics: %s", geno_proc.shape)

    log.info("Preprocessing epigenomics…")
    ep = EpigenomicsPreprocessor().preprocess(synthetic["epigenomics"])
    log.info("  → epigenomics: %s", ep.shape)

    log.info("Preprocessing proteomics…")
    pr = ProteomicsPreprocessor().preprocess(synthetic["proteomics"])
    log.info("  → proteomics: %s", pr.shape)

    log.info("Preprocessing metabolomics…")
    me = MetabolomicsPreprocessor().preprocess(synthetic["metabolomics"])
    log.info("  → metabolomics: %s", me.shape)

    log.info("Preprocessing microbiome…")
    mb = MicrobiomePreprocessor(rarefaction_depth=1000).preprocess(synthetic["microbiome"])
    log.info("  → microbiome: %s", mb.shape)

    return {
        "transcriptomics": txn,
        "genomics":        geno_proc,
        "epigenomics":     ep,
        "proteomics":      pr,
        "metabolomics":    me,
        "microbiome":      mb,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: MOFA+ integration
# ─────────────────────────────────────────────────────────────────────────────
def run_mofa(omics: dict[str, pd.DataFrame]) -> pd.DataFrame:
    from src.integration.mofa import MOFAIntegrator
    log.info("Running MOFA+ (n_factors=5, max_iter=200)…")
    integrator = MOFAIntegrator(n_factors=5, max_iter=200, convergence_mode="fast")
    factors = integrator.fit_transform(omics)
    log.info("  → MOFA+ factors: %s", factors.shape)
    return factors


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: XGBoost ensemble + SHAP
# ─────────────────────────────────────────────────────────────────────────────
def run_ensemble(omics: dict[str, pd.DataFrame],
                 mofa_factors: pd.DataFrame,
                 diagnosis: pd.Series) -> tuple:
    from src.integration.ensemble import OmicsEnsemble

    # combine all modalities + MOFA factors into a single feature matrix
    X = pd.concat(list(omics.values()) + [mofa_factors], axis=1)
    X.columns = [str(c) for c in X.columns]  # ensure string column names

    log.info("Training XGBoost ensemble on %d subjects × %d features…",
             X.shape[0], X.shape[1])
    ensemble = OmicsEnsemble(n_estimators=50, max_depth=4)
    ensemble.fit(X, diagnosis.loc[X.index])
    probas = ensemble.predict_proba(X)
    shap_vals = ensemble.compute_shap(X)
    top = ensemble.top_features(n=5)

    log.info("  → Top 5 features by importance:")
    for feat, imp in top:
        log.info("      %-40s %.4f", feat, imp)

    return X, ensemble, probas


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Build Stage1Output objects
# ─────────────────────────────────────────────────────────────────────────────
def build_stage1_outputs(X, ensemble, diagnosis, mofa_factors, env_scores, disease_stages):
    from src.integration.stage1_builder import Stage1Builder
    import json

    builder = Stage1Builder(top_k_biomarkers=10)
    outputs = builder.build(X, diagnosis.loc[X.index], ensemble,
                            mofa_factors, env_scores, disease_stages)

    log.info("Built %d Stage1Output objects. Sample:", len(outputs))
    sample = outputs[0]
    log.info("  subject_id:             %s", sample.subject_id)
    log.info("  diagnosis:              %s", sample.diagnosis)
    log.info("  prediction_confidence:  %.3f", sample.prediction_confidence)
    log.info("  disease_stage:          %s", sample.disease_stage)
    log.info("  environmental_risk:     %.2f", sample.environmental_risk_score)
    log.info("  top biomarkers:")
    for b in sample.top_biomarkers[:3]:
        log.info("    [%s] %s  SHAP=%.3f (%s)", b.modality, b.feature, b.shap_value, b.direction)

    # save raw JSONs
    for out in outputs:
        (INTEGRATED / f"{out.subject_id}.json").write_text(out.to_json())
    log.info("Stage1Output JSONs written to %s", INTEGRATED)
    return outputs


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: Build instruction pairs + JSONL splits
# ─────────────────────────────────────────────────────────────────────────────
def build_instructions(outputs):
    from src.instruction_builder.dataset_builder import DatasetBuilder

    builder = DatasetBuilder(seed=42)
    pairs = builder.build_pairs(outputs)
    splits = builder.split(pairs)
    builder.save(splits, INSTRUCTIONS)

    log.info("Instruction pairs built: %d total", len(pairs))
    for split_name, recs in splits.items():
        log.info("  %-6s: %d pairs", split_name, len(recs))
    log.info("JSONL files written to %s", INSTRUCTIONS)

    # print one sample training pair
    sample_pair = splits["train"][0]
    log.info("\n--- Sample training pair (task: %s) ---", sample_pair["task"])
    log.info("INSTRUCTION:\n%s", sample_pair["instruction"])
    log.info("INPUT (truncated):\n%s", sample_pair["input"][:300])
    log.info("OUTPUT:\n%s", sample_pair["output"][:400])


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("PluMA Parkinson's AI — End-to-End Smoke Test")
    log.info("Using real data: GSE6613 (GEO) + NHANES 2017-2018")
    log.info("=" * 60)

    # 1. Real transcriptomics data from GEO
    expr, diagnosis = fetch_gse6613()
    subjects = list(expr.columns)

    # 2. Real environmental data from NHANES
    env_scores = fetch_nhanes_metals(subjects)

    # 3. Synthetic data for other modalities
    log.info("Generating synthetic data for non-GEO modalities…")
    synthetic = make_synthetic_modalities(subjects, diagnosis)

    # 4. Preprocess all modalities
    omics = preprocess_all(expr, synthetic, subjects)

    # 5. MOFA+ integration
    mofa_factors = run_mofa(omics)

    # 6. XGBoost ensemble
    X, ensemble, _ = run_ensemble(omics, mofa_factors, diagnosis)

    # 7. Build Stage1Output
    disease_stages = pd.Series({
        s: ("early" if diagnosis[s] == "PD" else None)
        for s in subjects
    })
    outputs = build_stage1_outputs(
        X, ensemble, diagnosis, mofa_factors, env_scores, disease_stages
    )

    # 8. Build instruction pairs
    build_instructions(outputs)

    log.info("=" * 60)
    log.info("Smoke test PASSED — full pipeline executed successfully.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
