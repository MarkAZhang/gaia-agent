import pytest

from agent.prompts.get_prompt import get_prompt


def test_get_prompt_returns_react_system_prompt():
    prompt = get_prompt("react_system_prompt")
    assert "ReAct agent" in prompt
    assert "THINK" in prompt
    assert "ACT" in prompt
    assert "OBSERVE" in prompt


def test_get_prompt_strips_whitespace():
    prompt = get_prompt("react_system_prompt")
    assert not prompt.startswith("\n")
    assert not prompt.endswith("\n")


def test_get_prompt_raises_on_missing_key():
    with pytest.raises(KeyError, match="nonexistent"):
        get_prompt("nonexistent")
