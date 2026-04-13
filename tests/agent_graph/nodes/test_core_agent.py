from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from agent_graph.agent_dependencies import AgentDependencies
from agent_graph.nodes.core_agent import core_agent


def _make_config(mock_llm: MagicMock) -> RunnableConfig:
    return RunnableConfig(configurable={"deps": AgentDependencies(core_agent_model=mock_llm)})


def test_core_agent_passes_agent_messages_through():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Hello!")
    config = _make_config(mock_llm)

    state = {
        "agent_messages": [SystemMessage(content="sys"), HumanMessage(content="Hi")],
        "tool_messages": [],
    }
    result = core_agent(state, config)

    invoked_messages = mock_llm.invoke.call_args[0][0]
    assert invoked_messages == state["agent_messages"]
    assert result == {"agent_messages": [AIMessage(content="Hello!")]}


def test_core_agent_includes_latest_tool_message():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Got the result.")
    config = _make_config(mock_llm)

    tool_msg = ToolMessage(content="tool result", tool_call_id="1")
    state = {
        "agent_messages": [HumanMessage(content="Hi")],
        "tool_messages": [tool_msg],
    }
    result = core_agent(state, config)

    invoked_messages = mock_llm.invoke.call_args[0][0]
    assert len(invoked_messages) == 2
    assert invoked_messages[0] == HumanMessage(content="Hi")
    assert invoked_messages[1] == tool_msg
    assert result == {"agent_messages": [AIMessage(content="Got the result.")]}


def test_core_agent_does_not_inject_system_prompt():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Hello!")
    config = _make_config(mock_llm)

    state = {"agent_messages": [HumanMessage(content="Hi")], "tool_messages": []}
    core_agent(state, config)

    invoked_messages = mock_llm.invoke.call_args[0][0]
    assert len(invoked_messages) == 1
    assert invoked_messages[0] == HumanMessage(content="Hi")


def test_core_agent_returns_tool_calls():
    mock_llm = MagicMock()
    ai_msg = AIMessage(
        content="Let me use the tool.",
        tool_calls=[{"name": "noop_tool", "args": {"input": "hello"}, "id": "1"}],
    )
    mock_llm.invoke.return_value = ai_msg
    config = _make_config(mock_llm)

    state = {"agent_messages": [HumanMessage(content="Use noop")], "tool_messages": []}
    result = core_agent(state, config)

    assert result["agent_messages"][0].tool_calls == [
        {"name": "noop_tool", "args": {"input": "hello"}, "id": "1", "type": "tool_call"}
    ]
