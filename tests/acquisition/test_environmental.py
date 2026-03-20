# tests/acquisition/test_environmental.py
import pytest
from unittest.mock import patch, MagicMock
from src.acquisition.environmental import EPAClient, NHANESClient, ExposureRecord

def test_exposure_record_creation():
    r = ExposureRecord(subject_id="PD_001", pm25=8.2, pm10=15.1,
                       no2=12.3, ozone=45.0,
                       pesticide_score=3.4, heavy_metals_score=1.2)
    assert r.pm25 == 8.2
    assert r.subject_id == "PD_001"

def test_exposure_record_to_dict():
    r = ExposureRecord(subject_id="PD_001", pm25=8.2, pm10=15.1,
                       no2=12.3, ozone=45.0,
                       pesticide_score=3.4, heavy_metals_score=1.2)
    d = r.to_dict()
    assert d["subject_id"] == "PD_001"
    assert d["pm25"] == 8.2
    assert set(d.keys()) == {"subject_id", "pm25", "pm10", "no2", "ozone",
                              "pesticide_score", "heavy_metals_score"}

def test_epa_client_initializes(tmp_path):
    client = EPAClient(api_key="TEST_KEY", data_dir=str(tmp_path))
    assert client.data_dir == tmp_path
    assert client.api_key == "TEST_KEY"

def test_epa_client_builds_url():
    client = EPAClient(api_key="MYKEY", data_dir="data/raw/epa")
    url = client._build_url("dailyData/byCounty", param="88101",
                            bdate="20180101", edate="20181231",
                            state="06", county="037")
    assert "88101" in url
    assert "dailyData/byCounty" in url
    assert "MYKEY" in url
    assert url.startswith("https://aqs.epa.gov")

def test_nhanes_client_initializes(tmp_path):
    client = NHANESClient(data_dir=str(tmp_path))
    assert client.data_dir == tmp_path

def test_nhanes_known_cycles():
    client = NHANESClient(data_dir="data/raw/nhanes")
    assert "2017-2018" in client.EXPOSURE_FILES
    assert "2019-2020" in client.EXPOSURE_FILES
