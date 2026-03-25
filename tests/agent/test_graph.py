from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from agent.deps import AgentDeps
from agent.graph import build_graph


def _build_with_mock(mock_llm: MagicMock):
    config = RunnableConfig(configurable={"deps": AgentDeps(llm=mock_llm)})
    graph = build_graph()
    return graph, config


def test_graph_ends_when_no_tool_calls():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Hello!")
    graph, config = _build_with_mock(mock_llm)

    result = graph.invoke(
        {"messages": [HumanMessage(content="Hi")]},
        config=config,
    )

    messages = result["messages"]
    assert messages[-1].content == "Hello!"
    assert not messages[-1].tool_calls
    mock_llm.invoke.assert_called_once()


def test_graph_calls_tool_and_loops_back():
    mock_llm = MagicMock()
    # First call: LLM decides to use a tool
    tool_call_msg = AIMessage(
        content="Let me use the tool.",
        tool_calls=[{"name": "noop_tool", "args": {"input": "hello"}, "id": "call_1"}],
    )
    # Second call: LLM gives final answer after seeing tool result
    final_msg = AIMessage(content="The tool returned: hello to you too!")

    mock_llm.invoke.side_effect = [tool_call_msg, final_msg]
    graph, config = _build_with_mock(mock_llm)

    result = graph.invoke(
        {"messages": [HumanMessage(content="Use noop")]},
        config=config,
    )

    messages = result["messages"]
    assert mock_llm.invoke.call_count == 2

    # Verify the sequence: human -> ai (tool_call) -> tool -> ai (final)
    assert messages[0].type == "human"
    assert messages[1].type == "ai"
    assert messages[1].tool_calls
    assert messages[2].type == "tool"
    assert messages[3].type == "ai"
    assert messages[3].content == "The tool returned: hello to you too!"


def test_graph_handles_multiple_tool_calls():
    mock_llm = MagicMock()
    # First call: use tool
    first_tool_msg = AIMessage(
        content="First tool call.",
        tool_calls=[{"name": "noop_tool", "args": {"input": "a"}, "id": "call_1"}],
    )
    # Second call: use tool again
    second_tool_msg = AIMessage(
        content="Second tool call.",
        tool_calls=[{"name": "noop_tool", "args": {"input": "b"}, "id": "call_2"}],
    )
    # Third call: final answer
    final_msg = AIMessage(content="Done.")

    mock_llm.invoke.side_effect = [first_tool_msg, second_tool_msg, final_msg]
    graph, config = _build_with_mock(mock_llm)

    result = graph.invoke(
        {"messages": [HumanMessage(content="Use noop twice")]},
        config=config,
    )

    assert mock_llm.invoke.call_count == 3
    assert result["messages"][-1].content == "Done."


def test_graph_tool_result_content():
    mock_llm = MagicMock()
    tool_call_msg = AIMessage(
        content="Calling tool.",
        tool_calls=[{"name": "noop_tool", "args": {"input": "test"}, "id": "call_1"}],
    )
    final_msg = AIMessage(content="Got it.")

    mock_llm.invoke.side_effect = [tool_call_msg, final_msg]
    graph, config = _build_with_mock(mock_llm)

    result = graph.invoke(
        {"messages": [HumanMessage(content="Test")]},
        config=config,
    )

    # Find the tool message and verify the noop_tool returned the expected value
    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "test to you too!"
