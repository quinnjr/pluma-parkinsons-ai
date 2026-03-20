"""
QLoRA fine-tuning for Mistral Small 4 on PD multi-omics instruction data.

Usage:
    python -m src.training.train --config configs/training.yaml \
        --data_dir data/integrated/instruction_pairs

Requirements:
    pip install -e ".[training]"

Hardware:
    Minimum 20GB VRAM (RTX 3090/4090). For multi-GPU, use:
    accelerate launch -m src.training.train --config configs/training.yaml
"""
from __future__ import annotations
import argparse
from pathlib import Path
from src.utils import load_config
from src.training.model_utils import format_prompt


def load_model_and_tokenizer(cfg: dict):
    """Load Mistral Small 4 with 4-bit NF4 quantization and LoRA adapters."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["model"]["load_in_4bit"],
        bnb_4bit_quant_type=cfg["model"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=cfg["model"]["use_nested_quant"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        target_modules=cfg["lora"]["target_modules"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        bias=cfg["lora"]["bias"],
        task_type=cfg["lora"]["task_type"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["name"], trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def prepare_dataset(data_dir: str | Path, max_seq_length: int):
    """Load JSONL splits and apply Alpaca prompt template."""
    from datasets import load_dataset

    data_dir = Path(data_dir)
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "val.jsonl"),
        },
    )
    return dataset.map(lambda ex: {"text": format_prompt(ex)})


def train(config_path: str, data_dir: str) -> None:
    from transformers import TrainingArguments
    from trl import SFTTrainer

    cfg = load_config(config_path)
    model, tokenizer = load_model_and_tokenizer(cfg)
    dataset = prepare_dataset(data_dir, cfg["training"]["max_seq_length"])

    training_args = TrainingArguments(
        output_dir=cfg["training"]["output_dir"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        save_steps=cfg["training"]["save_steps"],
        logging_steps=cfg["training"]["logging_steps"],
        bf16=True,
        fp16=False,
        group_by_length=True,
        report_to="none",
        seed=cfg["training"]["seed"],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=cfg["training"]["max_seq_length"],
        packing=True,
    )

    trainer.train()
    trainer.save_model(cfg["training"]["output_dir"])

    if cfg["hub"].get("push_to_hub"):
        model.push_to_hub(cfg["hub"]["hub_model_id"])
        tokenizer.push_to_hub(cfg["hub"]["hub_model_id"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for PD multi-omics")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--data_dir", default="data/integrated/instruction_pairs")
    args = parser.parse_args()
    train(args.config, args.data_dir)
