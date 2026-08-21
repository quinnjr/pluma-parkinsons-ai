"""Conversation construction for Gemma 4.

Gemma 4 does not use the Alpaca ``### Instruction:`` layout. Its turns are
delimited by ``<|turn>role ... <turn|>`` markers and the model card ships a
Jinja chat template that also handles the thinking channel. Hand-writing that
format would drift the moment Google revises the template, so nothing here
emits turn markers: we build message lists and let the tokenizer render them.
"""
from __future__ import annotations

from typing import Any

from src.instruction_builder.templates import SYSTEM_PROMPT


def build_messages(pair: dict, include_completion: bool = True) -> dict[str, list[dict[str, str]]]:
    """Convert an instruction pair into TRL's prompt-completion conversational form.

    Returning ``prompt`` and ``completion`` separately (rather than one merged
    ``messages`` list) is what lets `SFTTrainer` mask the prompt tokens and
    compute loss on the response only.
    """
    user_content = pair["instruction"]
    context = (pair.get("input") or "").strip()
    if context:
        user_content = f"{user_content}\n\n{context}"

    record: dict[str, Any] = {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
    }
    if include_completion:
        record["completion"] = [{"role": "assistant", "content": pair["output"]}]
    return record


def render_for_inference(tokenizer, pair: dict) -> str:
    """Render the prompt exactly as it will appear at generation time."""
    messages = build_messages(pair, include_completion=False)["prompt"]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
