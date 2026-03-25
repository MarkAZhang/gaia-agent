from langchain_core.messages import AIMessage
from langgraph.graph import END

from agent.nodes.should_continue import should_continue


def test_should_continue_returns_tools_when_tool_calls_present():
    msg = AIMessage(
        content="Using tool.",
        tool_calls=[{"name": "noop_tool", "args": {"input": "hi"}, "id": "1"}],
    )
    state = {"messages": [msg]}
    assert should_continue(state) == "tools"


def test_should_continue_returns_end_when_no_tool_calls():
    msg = AIMessage(content="Final answer.")
    state = {"messages": [msg]}
    assert should_continue(state) == END


def test_should_continue_returns_end_with_empty_tool_calls():
    msg = AIMessage(content="Done.", tool_calls=[])
    state = {"messages": [msg]}
    assert should_continue(state) == END


def test_should_continue_uses_last_message():
    first = AIMessage(
        content="Using tool.",
        tool_calls=[{"name": "noop_tool", "args": {"input": "hi"}, "id": "1"}],
    )
    last = AIMessage(content="Final answer.")
    state = {"messages": [first, last]}
    assert should_continue(state) == END
