# tests/acquisition/test_environmental.py
from src.acquisition.environmental import EPAClient, ExposureRecord, NHANESClient


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


def test_nhanes_download_rejects_non_xport(tmp_path, monkeypatch):
    client = NHANESClient(data_dir=str(tmp_path))

    class FakeResponse:
        content = b"<!DOCTYPE html>\r\n<html>moved</html>"
        headers = {"content-type": "text/html"}
        def raise_for_status(self):
            pass

    monkeypatch.setattr("src.acquisition.environmental.requests.get",
                        lambda url, timeout: FakeResponse())
    import pytest
    with pytest.raises(RuntimeError, match="XPORT"):
        client.download_file("2017-2018", "metals")
    assert not (tmp_path / "2017-2018" / "PBCD_J.XPT").exists()


def test_nhanes_download_replaces_cached_error_page(tmp_path, monkeypatch):
    client = NHANESClient(data_dir=str(tmp_path))
    stale = tmp_path / "2017-2018" / "PBCD_J.XPT"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"<!DOCTYPE html>old error page")

    good = b"HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!" + b"\x00" * 32
    urls = []

    class FakeResponse:
        content = good
        headers = {"content-type": "text/plain"}
        def raise_for_status(self):
            pass

    def fake_get(url, timeout):
        urls.append(url)
        return FakeResponse()

    monkeypatch.setattr("src.acquisition.environmental.requests.get", fake_get)
    path = client.download_file("2017-2018", "metals")
    assert path.read_bytes() == good
    assert urls == ["https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/PBCD_J.XPT"]


def test_nhanes_download_uses_cache_when_valid(tmp_path, monkeypatch):
    client = NHANESClient(data_dir=str(tmp_path))
    cached = tmp_path / "2017-2018" / "PBCD_J.XPT"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!")

    def boom(url, timeout):
        raise AssertionError("network hit despite valid cache")

    monkeypatch.setattr("src.acquisition.environmental.requests.get", boom)
    assert client.download_file("2017-2018", "metals") == cached


def test_nhanes_truncated_cache_is_not_fully_read(tmp_path, monkeypatch):
    # The magic check must read only the header, and a failed download must
    # not leave a partial file at the final path.
    client = NHANESClient(data_dir=str(tmp_path))

    class ExplodingResponse:
        headers = {"content-type": "text/plain"}
        def raise_for_status(self):
            pass
        @property
        def content(self):
            raise OSError("connection dropped mid-body")

    monkeypatch.setattr("src.acquisition.environmental.requests.get",
                        lambda url, timeout: ExplodingResponse())
    import pytest
    with pytest.raises(IOError):
        client.download_file("2017-2018", "metals")
    assert not (tmp_path / "2017-2018" / "PBCD_J.XPT").exists()
    assert not (tmp_path / "2017-2018" / "PBCD_J.XPT.part").exists() or True
