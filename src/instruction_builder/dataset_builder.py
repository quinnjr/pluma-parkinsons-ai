from __future__ import annotations

import random
from pathlib import Path

from src.instruction_builder.formatter import InstructionFormatter
from src.models import Stage1Output
from src.utils import save_jsonl


class DatasetBuilder:
    """Build instruction pairs and split them into train/val/test JSONL files."""

    def __init__(self, train_frac: float = 0.8, val_frac: float = 0.1, seed: int = 42,
                 formatter: InstructionFormatter | None = None):
        if not 0 < train_frac < 1 or not 0 <= val_frac < 1 or train_frac + val_frac >= 1:
            raise ValueError(
                f"train_frac and val_frac must leave a non-empty test split; got "
                f"train={train_frac}, val={val_frac}"
            )
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.seed = seed
        self.formatter = formatter or InstructionFormatter(seed=seed)

    def build_pairs(self, outputs: list[Stage1Output]) -> list[dict]:
        """Convert each Stage1Output into one instruction-response pair per task."""
        pairs = []
        for output in outputs:
            pairs.extend(self.formatter.all_formats(output))
        return pairs

    def split(self, pairs: list[dict]) -> dict[str, list[dict]]:
        """Split by *subject*, not by pair.

        The three tasks generated for one subject share the same profile text
        verbatim. Splitting pairs at random puts near-identical inputs on both
        sides of the boundary, and the resulting validation loss measures
        memorisation rather than generalisation.
        """
        subjects = sorted({pair["subject_id"] for pair in pairs})
        rng = random.Random(self.seed)
        rng.shuffle(subjects)

        n = len(subjects)
        if n < 3:
            # With 1-2 subjects the arithmetic below leaves train or val empty,
            # and the trainer fails much later with an unrelated error.
            raise ValueError(
                f"need at least 3 subjects for a train/val/test split, got {n}"
            )
        # Integer truncation can starve any split; clamp so all three end up
        # non-empty. The final n_val clamp is what guarantees test >= 1 — a
        # bare max(n_train, 1) after the test-reserving min() could otherwise
        # hand train+val the whole cohort.
        n_train = max(min(int(n * self.train_frac), n - 2), 1)
        n_val = max(int(n * self.val_frac), 1)
        n_val = min(n_val, n - n_train - 1)

        assignment = {}
        for i, subject in enumerate(subjects):
            if i < n_train:
                assignment[subject] = "train"
            elif i < n_train + n_val:
                assignment[subject] = "val"
            else:
                assignment[subject] = "test"

        splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
        for pair in pairs:
            splits[assignment[pair["subject_id"]]].append(pair)
        return splits

    def save(self, splits: dict[str, list[dict]], output_dir: str | Path) -> None:
        """Write each split to a JSONL file."""
        output_dir = Path(output_dir)
        for split_name, records in splits.items():
            save_jsonl(records, output_dir / f"{split_name}.jsonl")
