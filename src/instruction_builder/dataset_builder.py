from __future__ import annotations
import random
from pathlib import Path
from src.models import Stage1Output
from src.instruction_builder.formatter import InstructionFormatter
from src.utils import save_jsonl


class DatasetBuilder:
    def __init__(self, train_frac: float = 0.8, val_frac: float = 0.1,
                 seed: int = 42):
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.seed = seed
        self.formatter = InstructionFormatter(seed=seed)

    def build_pairs(self, outputs: list[Stage1Output]) -> list[dict]:
        """Convert each Stage1Output into 3 instruction-response pairs."""
        pairs = []
        for output in outputs:
            pairs.extend(self.formatter.all_formats(output))
        return pairs

    def split(self, pairs: list[dict]) -> dict[str, list[dict]]:
        """80/10/10 train/val/test split with shuffling."""
        rng = random.Random(self.seed)
        shuffled = pairs.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * self.train_frac)
        n_val = int(n * self.val_frac)
        return {
            "train": shuffled[:n_train],
            "val": shuffled[n_train: n_train + n_val],
            "test": shuffled[n_train + n_val:],
        }

    def save(self, splits: dict[str, list[dict]], output_dir: str | Path) -> None:
        """Write each split to a JSONL file."""
        output_dir = Path(output_dir)
        for split_name, records in splits.items():
            save_jsonl(records, output_dir / f"{split_name}.jsonl")
