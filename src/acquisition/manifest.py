from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.models import VALID_DIAGNOSES


class SampleManifest:
    """Maps subject IDs across databases and tracks metadata."""

    def __init__(self):
        self._subjects: dict[str, dict] = {}
        self._external_ids: dict[tuple[str, str], str] = {}  # (source, ext_id) -> subject_id

    def add_subject(self, subject_id: str, diagnosis: str, age: int, sex: str, cohort: str) -> None:
        if diagnosis not in VALID_DIAGNOSES:
            raise ValueError(f"Invalid diagnosis: {diagnosis!r}. Must be one of {VALID_DIAGNOSES}")
        self._subjects[subject_id] = {
            "subject_id": subject_id,
            "diagnosis": diagnosis,
            "age": age,
            "sex": sex,
            "cohort": cohort,
        }

    def link_external_id(self, subject_id: str, source: str, external_id: str) -> None:
        if subject_id not in self._subjects:
            raise KeyError(f"Unknown subject: {subject_id!r}")
        self._external_ids[(source, external_id)] = subject_id

    def resolve_external(self, source: str, external_id: str) -> str | None:
        return self._external_ids.get((source, external_id))

    def get(self, subject_id: str) -> dict:
        return self._subjects[subject_id]

    def filter(self, **kwargs) -> list[str]:
        return [
            sid for sid, meta in self._subjects.items()
            if all(meta.get(k) == v for k, v in kwargs.items())
        ]

    def __contains__(self, subject_id: str) -> bool:
        return subject_id in self._subjects

    def __len__(self) -> int:
        return len(self._subjects)

    def save(self, path: str | Path) -> None:
        pd.DataFrame(self._subjects.values()).to_csv(path, index=False)

    @classmethod
    def load(cls, path: str | Path) -> SampleManifest:
        m = cls()
        for _, row in pd.read_csv(path).iterrows():
            m.add_subject(
                subject_id=row["subject_id"],
                diagnosis=row["diagnosis"],
                age=int(row["age"]),
                sex=row["sex"],
                cohort=row["cohort"],
            )
        return m
