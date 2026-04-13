from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, ToolMessage

from agent_graph.nodes.tool_node import tool_node


def _make_ai_msg_with_tool_calls(tool_calls):
    return AIMessage(content="Invoking tools.", tool_calls=tool_calls)


def test_tool_node_returns_tool_messages():
    tool_msg = ToolMessage(content="Result for: test", tool_call_id="call_1")
    mock_prebuilt = MagicMock()
    mock_prebuilt.invoke.return_value = {"messages": [tool_msg]}

    with patch("agent_graph.nodes.tool_node.ToolNode", return_value=mock_prebuilt):
        run_tools = tool_node([])

    ai_msg = _make_ai_msg_with_tool_calls(
        [{"name": "fake_tool", "args": {"query": "test"}, "id": "call_1"}]
    )
    state = {"agent_messages": [ai_msg], "tool_messages": []}

    result = run_tools(state)

    assert "tool_messages" in result
    assert len(result["tool_messages"]) == 1
    assert isinstance(result["tool_messages"][0], ToolMessage)
    assert "Result for: test" in result["tool_messages"][0].content
    mock_prebuilt.invoke.assert_called_once_with({"messages": [ai_msg]})


def test_tool_node_does_not_return_agent_messages():
    tool_msg = ToolMessage(content="result", tool_call_id="call_1")
    mock_prebuilt = MagicMock()
    mock_prebuilt.invoke.return_value = {"messages": [tool_msg]}

    with patch("agent_graph.nodes.tool_node.ToolNode", return_value=mock_prebuilt):
        run_tools = tool_node([])

    ai_msg = _make_ai_msg_with_tool_calls(
        [{"name": "fake_tool", "args": {"query": "hello"}, "id": "call_1"}]
    )
    state = {"agent_messages": [ai_msg], "tool_messages": []}

    result = run_tools(state)

    assert "agent_messages" not in result


def test_tool_node_handles_multiple_tool_calls():
    tool_msg_a = ToolMessage(content="Result for: a", tool_call_id="call_1")
    tool_msg_b = ToolMessage(content="Result for: b", tool_call_id="call_2")
    mock_prebuilt = MagicMock()
    mock_prebuilt.invoke.return_value = {"messages": [tool_msg_a, tool_msg_b]}

    with patch("agent_graph.nodes.tool_node.ToolNode", return_value=mock_prebuilt):
        run_tools = tool_node([])

    ai_msg = _make_ai_msg_with_tool_calls(
        [
            {"name": "fake_tool", "args": {"query": "a"}, "id": "call_1"},
            {"name": "fake_tool", "args": {"query": "b"}, "id": "call_2"},
        ]
    )
    state = {"agent_messages": [ai_msg], "tool_messages": []}

    result = run_tools(state)

    assert len(result["tool_messages"]) == 2
    assert "Result for: a" in result["tool_messages"][0].content
    assert "Result for: b" in result["tool_messages"][1].content


def test_tool_node_uses_last_agent_message():
    mock_prebuilt = MagicMock()
    mock_prebuilt.invoke.return_value = {"messages": []}

    with patch("agent_graph.nodes.tool_node.ToolNode", return_value=mock_prebuilt):
        run_tools = tool_node([])

    first_msg = AIMessage(content="First message.")
    last_msg = _make_ai_msg_with_tool_calls(
        [{"name": "fake_tool", "args": {"query": "x"}, "id": "call_1"}]
    )
    state = {"agent_messages": [first_msg, last_msg], "tool_messages": []}

    run_tools(state)

    mock_prebuilt.invoke.assert_called_once_with({"messages": [last_msg]})
