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


def load_model(cfg: dict, adapter: str | None):
    """Load the base model, optionally with a trained LoRA adapter attached.

    Quantization and tokenizer setup are the same config-driven helpers the
    training path uses — evaluating under a different quantization (or a
    tokenizer without the pad fallback) than the model was trained with would
    silently invalidate the metrics. The model class comes from the checkpoint
    config's own architecture: gemma-4-*-it is a ``gemma4_unified`` checkpoint,
    and forcing AutoModelForCausalLM either fails or builds a module tree the
    adapter's weight paths do not match (see the note in train.py).
    """
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    from src.training.train import _load_tokenizer, build_quantization_config

    model_name = cfg["model"]["name"]
    config = AutoConfig.from_pretrained(model_name)
    architectures = getattr(config, "architectures", None) or []
    model_cls = next(
        (getattr(transformers, arch) for arch in architectures
         if hasattr(transformers, arch)),
        AutoModelForCausalLM,
    )
    model = model_cls.from_pretrained(
        model_name,
        quantization_config=build_quantization_config(cfg),
        device_map="auto",
        dtype=getattr(torch, cfg["model"]["bnb_4bit_compute_dtype"]),
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    tokenizer = _load_tokenizer(cfg, source=adapter or model_name)
    return model, tokenizer


def generate(model, tokenizer, records: list[dict], max_new_tokens: int = 768,
             batch_size: int = 1) -> list[str]:
    """Greedy-decode a response for each record."""
    import torch

    outputs = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        prompts = [render_for_inference(tokenizer, record) for record in batch]
        # The chat template already emitted BOS; re-adding special tokens here
        # would prepend a second one, a prefix the model never saw in training.
        encoded = tokenizer(prompts, return_tensors="pt", padding=True,
                            padding_side="left", add_special_tokens=False)
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        # Explicit None check: Gemma's real <pad> token has id 0, which a bare
        # `or` would discard in favour of EOS.
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
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

    model, tokenizer = load_model(cfg, adapter)
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
