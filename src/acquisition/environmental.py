from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import requests
from src.utils import ensure_dir

EPA_BASE_URL = "https://aqs.epa.gov/data/api"

PM25_PARAM = "88101"
PM10_PARAM = "81102"
NO2_PARAM = "42602"
OZONE_PARAM = "44201"


@dataclass
class ExposureRecord:
    subject_id: str
    pm25: float
    pm10: float
    no2: float
    ozone: float
    pesticide_score: float
    heavy_metals_score: float

    def to_dict(self) -> dict:
        return asdict(self)


class EPAClient:
    """Fetches air quality data from EPA Air Quality System API."""

    def __init__(self, api_key: str, data_dir: str | Path):
        self.api_key = api_key
        self.data_dir = Path(data_dir)
        ensure_dir(self.data_dir)

    def _build_url(self, endpoint: str, **params) -> str:
        base = f"{EPA_BASE_URL}/{endpoint}?email=user@example.com&key={self.api_key}"
        for k, v in params.items():
            base += f"&{k}={v}"
        return base

    def fetch_county_annual(self, param: str, state: str, county: str,
                            year: int) -> "pd.DataFrame":
        import pandas as pd
        url = self._build_url(
            "annualData/byCounty",
            param=param,
            bdate=f"{year}0101",
            edate=f"{year}1231",
            state=state,
            county=county,
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("Data", [])
        return pd.DataFrame(data)


class NHANESClient:
    """Downloads NHANES environmental exposure data."""

    BASE_URL = "https://wwwn.cdc.gov/Nchs/Nhanes"
    EXPOSURE_FILES = {
        "2017-2018": {
            "metals": "PBCD_J.XPT",
            "pesticides": "BFRPOL_J.XPT",
        },
        "2019-2020": {
            "metals": "PBCD_K.XPT",
            "pesticides": "BFRPOL_K.XPT",
        },
    }

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        ensure_dir(self.data_dir)

    def download_file(self, cycle: str, category: str) -> Path:
        filename = self.EXPOSURE_FILES[cycle][category]
        url = f"{self.BASE_URL}/{cycle}/{filename}"
        dest = self.data_dir / cycle / filename
        ensure_dir(dest.parent)
        if not dest.exists():
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        return dest
