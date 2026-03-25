import pytest

from agent.prompts.get_prompt import get_prompt


def test_get_prompt_returns_react_system_prompt():
    prompt = get_prompt("react_system_prompt")
    assert "Thought" in prompt
    assert "Action" in prompt
    assert "Observation" in prompt
    assert "Ans:" in prompt


def test_get_prompt_strips_whitespace():
    prompt = get_prompt("react_system_prompt")
    assert not prompt.startswith("\n")
    assert not prompt.endswith("\n")


def test_get_prompt_returns_final_answer_format_error():
    prompt = get_prompt("final_answer_format_error")
    assert "Ans:" in prompt


def test_get_prompt_raises_on_missing_key():
    with pytest.raises(KeyError, match="nonexistent"):
        get_prompt("nonexistent")
