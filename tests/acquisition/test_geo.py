# tests/acquisition/test_geo.py
from src.acquisition.geo import GEOClient, GEOStudy


def test_geo_study_parses_accession():
    study = GEOStudy(accession="GSE123456", title="PD RNA-seq study",
                     organism="Homo sapiens", platform="GPL570", n_samples=48)
    assert study.accession == "GSE123456"
    assert study.n_samples == 48

def test_geo_client_initializes(tmp_path):
    client = GEOClient(data_dir=str(tmp_path))
    assert client.data_dir == tmp_path

def test_build_soft_url():
    client = GEOClient(data_dir="data/raw/geo")
    url = client._build_soft_url("GSE123456")
    assert "GSE123456" in url
    assert url.startswith("https://")
    assert "ftp.ncbi.nlm.nih.gov" in url

def test_filter_pd_studies():
    studies = [
        GEOStudy("GSE1", "Parkinson blood RNA", "Homo sapiens", "GPL570", 20),
        GEOStudy("GSE2", "Breast cancer study", "Homo sapiens", "GPL570", 30),
        GEOStudy("GSE3", "PD gut microbiome 16S", "Homo sapiens", "GPL16791", 15),
        GEOStudy("GSE4", "alpha-synuclein aggregation in neurons", "Homo sapiens", "GPL570", 12),
    ]
    client = GEOClient(data_dir="data/raw/geo")
    filtered = client.filter_pd_studies(studies)
    assert len(filtered) == 3
    assert all(s.accession != "GSE2" for s in filtered)

def test_filter_is_case_insensitive():
    studies = [
        GEOStudy("GSE1", "PARKINSON'S DISEASE transcriptome", "Homo sapiens", "GPL570", 10),
    ]
    client = GEOClient(data_dir="data/raw/geo")
    filtered = client.filter_pd_studies(studies)
    assert len(filtered) == 1


def test_collapse_to_genes_handles_duplicate_probe_ids():
    import pandas as pd

    from src.acquisition.geo import GEOClient

    expr = pd.DataFrame(
        [[1.0, 2.0], [5.0, 9.0], [3.0, 3.1]],
        index=["p1", "p1", "p2"],
        columns=["s1", "s2"],
    )
    collapsed = GEOClient.collapse_to_genes(expr, {"p1": "GENE_A", "p2": "GENE_B"})
    assert list(collapsed.index) == ["GENE_B", "GENE_A"] or \
           sorted(collapsed.index) == ["GENE_A", "GENE_B"]
    # The first p1 row is kept; the duplicate must not cartesian-expand.
    assert collapsed.loc["GENE_A"].tolist() == [1.0, 2.0]
