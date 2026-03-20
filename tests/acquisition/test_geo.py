# tests/acquisition/test_geo.py
import pytest
from unittest.mock import patch, MagicMock
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
