import os
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

OPENAI_ENV = {"OPENAI_API_KEY": "test-key"}

from agent_graph.build_agent_graph_and_config import build_agent_graph_and_config


@tool
def fake_tool(query: str) -> str:
    """A fake tool that returns a canned response for testing."""
    return f"Result for: {query}"


def _mock_openai_response(content: str) -> MagicMock:
    mock_message = MagicMock()
    mock_message.content = content
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


def _build_with_mock(mock_llm: MagicMock):
    with (
        patch(
            "agent_graph.build_agent_graph_and_config.ChatAnthropic"
        ) as mock_chat_anthropic,
        patch(
            "agent_graph.build_agent_graph_and_config._get_tools",
            return_value=[fake_tool],
        ),
    ):
        mock_chat_anthropic.return_value.bind_tools.return_value = mock_llm

        result = build_agent_graph_and_config(langfuse_handler=None)
        return result.graph, result.config


@patch.dict(os.environ, OPENAI_ENV)
@patch("agent_graph.nodes.output_formatter.OpenAI")
def test_graph_ends_when_answer_formatted_correctly(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_openai_response("Hello!")

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


@patch.dict(os.environ, OPENAI_ENV)
@patch("agent_graph.nodes.output_formatter.OpenAI")
def test_graph_retries_when_answer_not_formatted(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_openai_response("42")

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


@patch.dict(os.environ, OPENAI_ENV)
@patch("agent_graph.nodes.output_formatter.OpenAI")
def test_graph_calls_tool_and_loops_back(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_openai_response(
        "LangGraph is a framework."
    )

    mock_llm = MagicMock()
    tool_call_msg = AIMessage(
        content="Let me search for that.",
        tool_calls=[
            {
                "name": "fake_tool",
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


@patch.dict(os.environ, OPENAI_ENV)
@patch("agent_graph.nodes.output_formatter.OpenAI")
def test_graph_handles_multiple_tool_calls(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_openai_response("Done.")

    mock_llm = MagicMock()
    first_tool_msg = AIMessage(
        content="First search.",
        tool_calls=[
            {
                "name": "fake_tool",
                "args": {"query": "topic a"},
                "id": "call_1",
            }
        ],
    )
    second_tool_msg = AIMessage(
        content="Second search.",
        tool_calls=[
            {
                "name": "fake_tool",
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


@patch.dict(os.environ, OPENAI_ENV)
@patch("agent_graph.nodes.output_formatter.OpenAI")
def test_graph_memory_management_removes_old_tool_messages(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_openai_response("Done.")

    mock_llm = MagicMock()
    first_tool_msg = AIMessage(
        content="First search.",
        tool_calls=[
            {
                "name": "fake_tool",
                "args": {"query": "topic a"},
                "id": "call_1",
            }
        ],
    )
    second_tool_msg = AIMessage(
        content="Second search.",
        tool_calls=[
            {
                "name": "fake_tool",
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

    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert len(tool_messages) == 2
    # First tool message should have been replaced by memory management
    assert tool_messages[0].content == "removed"
    # Second tool message should still have its original content
    assert "Result for: topic b" in tool_messages[1].content


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


@patch.dict(os.environ, OPENAI_ENV)
@patch("agent_graph.nodes.output_formatter.OpenAI")
def test_graph_tool_result_content(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_openai_response("Got it.")

    mock_llm = MagicMock()
    tool_call_msg = AIMessage(
        content="Searching.",
        tool_calls=[
            {
                "name": "fake_tool",
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
    assert "Result for: test" in tool_messages[0].content
