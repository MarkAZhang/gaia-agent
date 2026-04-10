import pytest

from agent_graph.prompts.get_prompt import get_prompt


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


def test_get_prompt_concatenates_nested_sections():
    prompt = get_prompt("react_system_prompt")
    # Content from multiple sections should all appear
    assert "Thought-Action-Observation" in prompt
    assert "Tool Usage" in prompt
    assert "Tool not available:" in prompt
    assert "Formatting rules" in prompt
    assert "Now please answer the following question:" in prompt


def test_get_prompt_formats_with_kwargs(monkeypatch):
    from agent_graph.prompts import get_prompt as module

    fake = {
        "greeting": "Hello {name}, welcome to {place}.",
        "nested": {"a": "Part A {x}", "b": "Part B {y}"},
    }
    monkeypatch.setattr(module, "_PROMPTS", fake)

    assert module.get_prompt("greeting", name="Mark", place="GAIA") == (
        "Hello Mark, welcome to GAIA."
    )
    result = module.get_prompt("nested", x="1", y="2")
    assert "Part A 1" in result
    assert "Part B 2" in result


def test_get_prompt_missing_format_key_raises(monkeypatch):
    from agent_graph.prompts import get_prompt as module

    monkeypatch.setattr(module, "_PROMPTS", {"g": "Hello {name}"})
    with pytest.raises(KeyError):
        module.get_prompt("g", other="x")
