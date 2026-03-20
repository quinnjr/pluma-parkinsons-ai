from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.acquisition.manifest import SampleManifest
from src.utils import ensure_dir

DIAGNOSIS_MAP = {
    "Parkinson's Disease": "PD",
    "Healthy Control": "HC",
    "SWEDD": "SWEDD",
    "Prodromal": "Prodromal",
    "PD": "PD",
    "HC": "HC",
}

SEX_MAP = {"Male": "M", "Female": "F", "M": "M", "F": "F"}


class PPMIClient:
    """
    Client for PPMI data files downloaded from ppmi-info.org.

    PPMI requires manual registration and download. Place downloaded
    CSV files in data_dir before running. This client parses the standard
    PPMI CSV exports (Demographics, genomics, proteomics, etc.) into the
    project's internal data models.
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        ensure_dir(self.data_dir)

    def _map_diagnosis(self, raw: str) -> str:
        if raw in DIAGNOSIS_MAP:
            return DIAGNOSIS_MAP[raw]
        raise ValueError(f"Unknown PPMI diagnosis label: {raw!r}")

    def _map_sex(self, raw: str) -> str:
        return SEX_MAP.get(str(raw), "M")

    def build_manifest_from_csv(self, demographics_csv: str | Path) -> SampleManifest:
        """Parse PPMI Demographics.csv into a SampleManifest."""
        df = pd.read_csv(demographics_csv, dtype={"PATNO": str})
        manifest = SampleManifest()
        for _, row in df.iterrows():
            subject_id = f"PPMI_{row['PATNO']}"
            diagnosis = self._map_diagnosis(str(row["COHORT_DEFINITION"]))
            age = int(row.get("AGE_AT_VISIT", 0))
            sex = self._map_sex(str(row.get("SEX", "M")))
            manifest.add_subject(
                subject_id=subject_id,
                diagnosis=diagnosis,
                age=age,
                sex=sex,
                cohort="PPMI",
            )
        return manifest

    def list_available_files(self) -> list[Path]:
        return sorted(self.data_dir.glob("**/*.csv"))
