"""Evaluate a fine-tuned adapter on the held-out instruction split.

    python -m src.training.evaluate --adapter models/gemma-4-12b-pd-multiomics \
        --data data/instructions/test.jsonl --limit 50

Reports token overlap against the reference response, how often the model
invents a PMID the knowledge base does not contain, how many of the pipeline's
ranked biomarkers it recovers, and diagnosis agreement on the prediction task.
Pass ``--base-only`` to score the base model for comparison; a fine-tune that
does not beat it is not worth shipping.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.training.model_utils import compute_metrics
from src.training.prompts import render_for_inference
from src.utils import load_config, load_jsonl

logger = logging.getLogger(__name__)


def load_model(model_name: str, adapter: str | None, load_in_4bit: bool = True):
    """Load the base model, optionally with a trained LoRA adapter attached."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quantization = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        if load_in_4bit
        else None
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quantization, device_map="auto", dtype=torch.bfloat16
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(adapter or model_name)
    return model, tokenizer


def generate(model, tokenizer, records: list[dict], max_new_tokens: int = 768,
             batch_size: int = 1) -> list[str]:
    """Greedy-decode a response for each record."""
    import torch

    outputs = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        prompts = [render_for_inference(tokenizer, record) for record in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        prompt_length = encoded["input_ids"].shape[1]
        outputs.extend(
            tokenizer.decode(row[prompt_length:], skip_special_tokens=True) for row in generated
        )
        logger.info("generated %d/%d", min(start + batch_size, len(records)), len(records))
    return outputs


def evaluate(config_path: str, data_path: str, adapter: str | None, limit: int | None,
             max_new_tokens: int, batch_size: int) -> dict:
    cfg = load_config(config_path)
    records = load_jsonl(data_path)
    if not records:
        raise ValueError(f"No records in {data_path}")
    if limit:
        records = records[:limit]

    model, tokenizer = load_model(cfg["model"]["name"], adapter)
    predictions = generate(model, tokenizer, records, max_new_tokens, batch_size)
    references = [record["output"] for record in records]
    return compute_metrics(predictions, references, records)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a PD multi-omics adapter")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--data", default="data/instructions/test.jsonl")
    parser.add_argument("--adapter", default=None, help="Path to the trained LoRA adapter")
    parser.add_argument("--base-only", action="store_true",
                        help="Score the base model with no adapter, as a baseline")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--out", default=None, help="Write metrics JSON here")
    args = parser.parse_args(argv)

    adapter = None if args.base_only else args.adapter
    if adapter is None and not args.base_only:
        parser.error("pass --adapter <path>, or --base-only to score the base model")

    metrics = evaluate(args.config, args.data, adapter, args.limit,
                       args.max_new_tokens, args.batch_size)
    print(json.dumps(metrics, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
