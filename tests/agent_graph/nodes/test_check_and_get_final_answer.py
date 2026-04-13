from langchain_core.messages import AIMessage

from agent_graph.nodes.check_and_get_final_answer import check_and_get_final_answer


def test_extracts_answer_when_last_line_has_ans_prefix():
    msg = AIMessage(content="Some reasoning.\nAns: 42")
    state = {"messages": [msg]}
    result = check_and_get_final_answer(state)
    assert result["messages"][0]["content"] == "42"


def test_extracts_answer_with_extra_whitespace():
    msg = AIMessage(content="Reasoning.\nAns:   hello world   ")
    state = {"messages": [msg]}
    result = check_and_get_final_answer(state)
    assert result["messages"][0]["content"] == "hello world"


def test_returns_error_message_when_no_ans_prefix():
    msg = AIMessage(content="I think the answer is 42.")
    state = {"messages": [msg]}
    result = check_and_get_final_answer(state)
    assert result["messages"][0]["role"] == "system"
    assert "Ans:" in result["messages"][0]["content"]


def test_returns_error_for_empty_content():
    msg = AIMessage(content="")
    state = {"messages": [msg]}
    result = check_and_get_final_answer(state)
    assert result["messages"][0]["role"] == "system"


def test_extracts_answer_when_only_ans_line():
    msg = AIMessage(content="Ans: Mars")
    state = {"messages": [msg]}
    result = check_and_get_final_answer(state)
    assert result["messages"][0]["content"] == "Mars"


def test_uses_last_line_only():
    msg = AIMessage(content="Ans: wrong\nMore reasoning.\nAns: correct")
    state = {"messages": [msg]}
    result = check_and_get_final_answer(state)
    assert result["messages"][0]["content"] == "correct"


def test_returns_ai_message_on_success():
    msg = AIMessage(content="Thinking...\nAns: 5")
    state = {"messages": [msg]}
    result = check_and_get_final_answer(state)
    assert result["messages"][0]["role"] == "ai"
