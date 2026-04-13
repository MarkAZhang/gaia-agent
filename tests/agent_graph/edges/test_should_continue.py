from langchain_core.messages import AIMessage

from agent_graph.edges.should_continue import should_continue


def test_should_continue_returns_tools_when_tool_calls_present():
    msg = AIMessage(
        content="Using tool.",
        tool_calls=[{"name": "noop_tool", "args": {"input": "hi"}, "id": "1"}],
    )
    state = {"agent_messages": [msg]}
    assert should_continue(state) == "tools"


def test_should_continue_returns_check_node_when_no_tool_calls():
    msg = AIMessage(content="Final answer.")
    state = {"agent_messages": [msg]}
    assert should_continue(state) == "check_and_get_final_answer"


def test_should_continue_returns_check_node_with_empty_tool_calls():
    msg = AIMessage(content="Done.", tool_calls=[])
    state = {"agent_messages": [msg]}
    assert should_continue(state) == "check_and_get_final_answer"


def test_should_continue_uses_last_message():
    first = AIMessage(
        content="Using tool.",
        tool_calls=[{"name": "noop_tool", "args": {"input": "hi"}, "id": "1"}],
    )
    last = AIMessage(content="Final answer.")
    state = {"agent_messages": [first, last]}
    assert should_continue(state) == "check_and_get_final_answer"


def test_should_continue_returns_refusal_when_stop_reason_is_refusal():
    msg = AIMessage(
        content="I cannot assist with that.",
        response_metadata={"stop_reason": "refusal"},
    )
    state = {"agent_messages": [msg]}
    assert should_continue(state) == "return_llm_refusal"


def test_should_continue_refusal_takes_priority_over_tool_calls():
    msg = AIMessage(
        content="Refused.",
        tool_calls=[{"name": "noop_tool", "args": {"input": "hi"}, "id": "1"}],
        response_metadata={"stop_reason": "refusal"},
    )
    state = {"agent_messages": [msg]}
    assert should_continue(state) == "return_llm_refusal"


def test_should_continue_ignores_non_refusal_stop_reason():
    msg = AIMessage(
        content="Done.",
        response_metadata={"stop_reason": "end_turn"},
    )
    state = {"agent_messages": [msg]}
    assert should_continue(state) == "check_and_get_final_answer"


def test_should_continue_returns_tool_not_available_when_last_line_matches():
    msg = AIMessage(
        content="I tried to find a tool.\nTool not available: No calculator found"
    )
    state = {"agent_messages": [msg]}
    assert should_continue(state) == "return_llm_tool_not_available"


def test_should_continue_returns_tool_not_available_single_line():
    msg = AIMessage(content="Tool not available: No matching tool")
    state = {"agent_messages": [msg]}
    assert should_continue(state) == "return_llm_tool_not_available"


def test_should_continue_tool_not_available_only_checks_last_line():
    msg = AIMessage(
        content="Tool not available: something\nAns: 42"
    )
    state = {"agent_messages": [msg]}
    assert should_continue(state) == "check_and_get_final_answer"


def test_should_continue_refusal_takes_priority_over_tool_not_available():
    msg = AIMessage(
        content="Tool not available: something",
        response_metadata={"stop_reason": "refusal"},
    )
    state = {"agent_messages": [msg]}
    assert should_continue(state) == "return_llm_refusal"
