"""QLoRA fine-tuning of Gemma 4 on PD multi-omics instruction data.

Usage:
    python -m src.training.train --config configs/training.yaml \
        --data_dir data/instructions

Requirements:
    pip install -e ".[training]"

Hardware:
    google/gemma-4-12B-it in 4-bit NF4 needs roughly 12 GB of VRAM for LoRA
    training at max_length 4096. The 31B dense variant needs ~28 GB. For
    multi-GPU:
        accelerate launch -m src.training.train --config configs/training.yaml

Note on the model class: gemma-4-*-it checkpoints are `gemma4_unified`
(text+vision+audio). Passing the model *id* to SFTTrainer lets TRL instantiate
the architecture named in the checkpoint config rather than forcing
AutoModelForCausalLM, and passing a plain tokenizer as `processing_class` keeps
the run on the text-only path.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.training.prompts import build_messages
from src.utils import load_config

logger = logging.getLogger(__name__)


def build_quantization_config(cfg: dict):
    from transformers import BitsAndBytesConfig

    model_cfg = cfg["model"]
    if not model_cfg.get("load_in_4bit", True):
        return None
    import torch

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, model_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=model_cfg["use_nested_quant"],
    )


def build_peft_config(cfg: dict):
    from peft import LoraConfig

    lora = cfg["lora"]
    return LoraConfig(
        r=lora["r"],
        lora_alpha=lora["lora_alpha"],
        target_modules=list(lora["target_modules"]),
        lora_dropout=lora["lora_dropout"],
        bias=lora["bias"],
        task_type=lora["task_type"],
    )


def build_sft_config(cfg: dict):
    from trl import SFTConfig

    training = cfg["training"]
    return SFTConfig(
        output_dir=training["output_dir"],
        num_train_epochs=training["num_train_epochs"],
        per_device_train_batch_size=training["per_device_train_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        learning_rate=training["learning_rate"],
        warmup_ratio=training["warmup_ratio"],
        lr_scheduler_type=training["lr_scheduler_type"],
        save_steps=training["save_steps"],
        logging_steps=training["logging_steps"],
        eval_strategy=training.get("eval_strategy", "steps"),
        eval_steps=training.get("eval_steps", training["save_steps"]),
        max_length=training["max_length"],
        packing=training.get("packing", False),
        # Prompt-completion data: loss on the response only. TRL infers this,
        # but stating it means a change in TRL's default cannot silently start
        # training the model to reproduce its own inputs.
        completion_only_loss=True,
        gradient_checkpointing=training.get("gradient_checkpointing", True),
        bf16=True,
        fp16=False,
        report_to=training.get("report_to", "none"),
        seed=training["seed"],
        model_init_kwargs={"dtype": "bfloat16"},
    )


def prepare_dataset(data_dir: str | Path):
    """Load the JSONL splits and convert them to prompt-completion conversations."""
    from datasets import load_dataset

    data_dir = Path(data_dir)
    files = {"train": data_dir / "train.jsonl", "validation": data_dir / "val.jsonl"}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing instruction data: {missing}. Run "
            f"`python -m src.pipeline --stage build_instructions` first."
        )
    dataset = load_dataset("json", data_files={k: str(v) for k, v in files.items()})
    # Drop the analysis-only columns; SFTTrainer expects prompt/completion alone.
    return dataset.map(build_messages, remove_columns=dataset["train"].column_names)


def train(config_path: str, data_dir: str) -> None:
    from trl import SFTTrainer

    cfg = load_config(config_path)
    dataset = prepare_dataset(data_dir)
    logger.info(
        "Training on %d pairs, validating on %d",
        len(dataset["train"]), len(dataset["validation"]),
    )

    tokenizer = _load_tokenizer(cfg)
    trainer = SFTTrainer(
        model=cfg["model"]["name"],
        args=build_sft_config(cfg),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        quantization_config=build_quantization_config(cfg),
        peft_config=build_peft_config(cfg),
    )
    trainer.train()
    trainer.save_model(cfg["training"]["output_dir"])
    tokenizer.save_pretrained(cfg["training"]["output_dir"])

    hub_cfg = cfg.get("hub") or {}
    if hub_cfg.get("push_to_hub"):
        trainer.push_to_hub(hub_cfg["hub_model_id"])


def _load_tokenizer(cfg: dict):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    # Gemma 4 ships a real <pad> token; falling back to EOS would make the
    # collator mask genuine end-of-turn tokens out of the loss.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for PD multi-omics")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--data_dir", default="data/instructions")
    args = parser.parse_args(argv)
    train(args.config, args.data_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
