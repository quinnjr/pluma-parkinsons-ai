from src.instruction_builder.templates import SYSTEM_PROMPT
from src.training.prompts import build_messages, render_for_inference

PAIR = {
    "instruction": "Identify biomarkers in this profile.",
    "input": "Subject: PD_001\nLRRK2: SHAP=+0.340",
    "output": "The top-ranked feature is LRRK2.",
}


def test_build_messages_separates_prompt_and_completion():
    record = build_messages(PAIR)
    assert [m["role"] for m in record["prompt"]] == ["system", "user"]
    assert record["completion"] == [
        {"role": "assistant", "content": "The top-ranked feature is LRRK2."}
    ]


def test_build_messages_uses_system_prompt():
    record = build_messages(PAIR)
    assert record["prompt"][0]["content"] == SYSTEM_PROMPT


def test_build_messages_merges_instruction_and_input():
    user = build_messages(PAIR)["prompt"][1]["content"]
    assert user.startswith("Identify biomarkers")
    assert "LRRK2: SHAP=+0.340" in user


def test_build_messages_without_input():
    pair = {"instruction": "What is PD?", "output": "A disease."}
    user = build_messages(pair)["prompt"][1]["content"]
    assert user == "What is PD?"


def test_build_messages_without_completion():
    record = build_messages(PAIR, include_completion=False)
    assert "completion" not in record


def test_no_handrolled_turn_markers():
    record = build_messages(PAIR)
    text = " ".join(m["content"] for m in record["prompt"] + record["completion"])
    assert "<|turn>" not in text
    assert "### Instruction:" not in text


class _StubTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return "\n".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"


def test_render_for_inference_delegates_to_chat_template():
    rendered = render_for_inference(_StubTokenizer(), PAIR)
    assert rendered.endswith("<assistant>")
    assert "Identify biomarkers" in rendered
    assert "The top-ranked feature" not in rendered
