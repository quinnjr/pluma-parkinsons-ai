from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import requests

from src.utils import ensure_dir

if TYPE_CHECKING:
    import pandas as pd

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
                            year: int) -> pd.DataFrame:
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

    #: CDC reorganised NHANES hosting; data files now live under
    #: /Nchs/Data/Nhanes/Public/<first-year-of-cycle>/DataFiles/. The old
    #: /Nchs/Nhanes/<cycle>/ URLs return an HTML notice page with HTTP 200.
    BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"
    #: SAS transport (XPORT) files begin with this fixed 80-byte library header.
    XPORT_MAGIC = b"HEADER RECORD"
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
        year = cycle.split("-")[0]
        url = f"{self.BASE_URL}/{year}/DataFiles/{filename}"
        dest = self.data_dir / cycle / filename
        ensure_dir(dest.parent)
        if dest.exists() and not self._is_xport(dest.read_bytes()):
            # A cached error page from the pre-move URL; refetch.
            dest.unlink()
        if not dest.exists():
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            if not self._is_xport(resp.content):
                raise RuntimeError(
                    f"{url} did not return a SAS XPORT file (got "
                    f"{resp.headers.get('content-type', 'unknown type')!r}); "
                    f"the CDC may have moved the file again"
                )
            dest.write_bytes(resp.content)
        return dest

    @classmethod
    def _is_xport(cls, content: bytes) -> bool:
        return content.startswith(cls.XPORT_MAGIC)
