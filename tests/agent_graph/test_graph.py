from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from agent_graph.agent_dependencies import AgentDependencies
from agent_graph.build_agent_graph_and_config import build_graph


@tool
def fake_tavily_search(query: str) -> str:
    """A fake search tool that returns a canned response for testing."""
    return f"Search results for: {query}"


def _build_with_mock(mock_llm: MagicMock):
    config = RunnableConfig(configurable={"deps": AgentDependencies(core_agent_model=mock_llm)})
    graph = build_graph(tools=[fake_tavily_search])
    return graph, config


def test_graph_ends_when_answer_formatted_correctly():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Thinking.\nAns: Hello!")
    graph, config = _build_with_mock(mock_llm)

    result = graph.invoke(
        {"messages": [HumanMessage(content="Hi")]},
        config=config,
    )

    messages = result["messages"]
    assert messages[-1].content == "Hello!"
    mock_llm.invoke.assert_called_once()


def test_graph_retries_when_answer_not_formatted():
    mock_llm = MagicMock()
    bad_msg = AIMessage(content="The answer is 42.")
    good_msg = AIMessage(content="Let me fix that.\nAns: 42")

    mock_llm.invoke.side_effect = [bad_msg, good_msg]
    graph, config = _build_with_mock(mock_llm)

    result = graph.invoke(
        {"messages": [HumanMessage(content="What is 6*7?")]},
        config=config,
    )

    assert mock_llm.invoke.call_count == 2
    assert result["messages"][-1].content == "42"


def test_graph_calls_tool_and_loops_back():
    mock_llm = MagicMock()
    tool_call_msg = AIMessage(
        content="Let me search for that.",
        tool_calls=[
            {
                "name": "fake_tavily_search",
                "args": {"query": "LangGraph"},
                "id": "call_1",
            }
        ],
    )
    final_msg = AIMessage(content="Ans: LangGraph is a framework.")

    mock_llm.invoke.side_effect = [tool_call_msg, final_msg]
    graph, config = _build_with_mock(mock_llm)

    result = graph.invoke(
        {"messages": [HumanMessage(content="Search for LangGraph")]},
        config=config,
    )

    messages = result["messages"]
    assert mock_llm.invoke.call_count == 2

    assert messages[0].type == "human"
    assert messages[1].type == "ai"
    assert messages[1].tool_calls
    assert messages[2].type == "tool"
    assert messages[3].type == "ai"
    assert messages[-1].content == "LangGraph is a framework."


def test_graph_handles_multiple_tool_calls():
    mock_llm = MagicMock()
    first_tool_msg = AIMessage(
        content="First search.",
        tool_calls=[
            {
                "name": "fake_tavily_search",
                "args": {"query": "topic a"},
                "id": "call_1",
            }
        ],
    )
    second_tool_msg = AIMessage(
        content="Second search.",
        tool_calls=[
            {
                "name": "fake_tavily_search",
                "args": {"query": "topic b"},
                "id": "call_2",
            }
        ],
    )
    final_msg = AIMessage(content="Ans: Done.")

    mock_llm.invoke.side_effect = [first_tool_msg, second_tool_msg, final_msg]
    graph, config = _build_with_mock(mock_llm)

    result = graph.invoke(
        {"messages": [HumanMessage(content="Search for two topics")]},
        config=config,
    )

    assert mock_llm.invoke.call_count == 3
    assert result["messages"][-1].content == "Done."


def test_graph_ends_with_refusal_message_when_llm_refuses():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content="I cannot assist with that.",
        response_metadata={"stop_reason": "refusal"},
    )
    graph, config = _build_with_mock(mock_llm)

    result = graph.invoke(
        {"messages": [HumanMessage(content="Do something bad")]},
        config=config,
    )

    messages = result["messages"]
    assert messages[-1].content == "LLM refused to continue"
    mock_llm.invoke.assert_called_once()


def test_graph_ends_with_tool_not_available_message():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content="I need a calculator.\nTool not available: No calculator tool found"
    )
    graph, config = _build_with_mock(mock_llm)

    result = graph.invoke(
        {"messages": [HumanMessage(content="What is 2+2?")]},
        config=config,
    )

    messages = result["messages"]
    assert messages[-1].content == "Tool not available: No calculator tool found"
    mock_llm.invoke.assert_called_once()


def test_graph_tool_result_content():
    mock_llm = MagicMock()
    tool_call_msg = AIMessage(
        content="Searching.",
        tool_calls=[
            {
                "name": "fake_tavily_search",
                "args": {"query": "test"},
                "id": "call_1",
            }
        ],
    )
    final_msg = AIMessage(content="Ans: Got it.")

    mock_llm.invoke.side_effect = [tool_call_msg, final_msg]
    graph, config = _build_with_mock(mock_llm)

    result = graph.invoke(
        {"messages": [HumanMessage(content="Test")]},
        config=config,
    )

    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert len(tool_messages) == 1
    assert "Search results for: test" in tool_messages[0].content
