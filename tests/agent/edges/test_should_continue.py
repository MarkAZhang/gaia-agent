from langchain_core.messages import AIMessage

from agent.edges.should_continue import should_continue


def test_should_continue_returns_tools_when_tool_calls_present():
    msg = AIMessage(
        content="Using tool.",
        tool_calls=[{"name": "noop_tool", "args": {"input": "hi"}, "id": "1"}],
    )
    state = {"messages": [msg]}
    assert should_continue(state) == "tools"


def test_should_continue_returns_check_node_when_no_tool_calls():
    msg = AIMessage(content="Final answer.")
    state = {"messages": [msg]}
    assert should_continue(state) == "check_and_get_final_answer"


def test_should_continue_returns_check_node_with_empty_tool_calls():
    msg = AIMessage(content="Done.", tool_calls=[])
    state = {"messages": [msg]}
    assert should_continue(state) == "check_and_get_final_answer"


def test_should_continue_uses_last_message():
    first = AIMessage(
        content="Using tool.",
        tool_calls=[{"name": "noop_tool", "args": {"input": "hi"}, "id": "1"}],
    )
    last = AIMessage(content="Final answer.")
    state = {"messages": [first, last]}
    assert should_continue(state) == "check_and_get_final_answer"
