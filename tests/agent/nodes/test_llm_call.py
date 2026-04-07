from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from agent.deps import AgentDeps
from agent.nodes.llm_call import llm_call


def _make_config(mock_llm: MagicMock) -> RunnableConfig:
    return RunnableConfig(configurable={"deps": AgentDeps(llm=mock_llm)})


def test_llm_call_passes_messages_through():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Hello!")
    config = _make_config(mock_llm)

    state = {
        "messages": [SystemMessage(content="sys"), HumanMessage(content="Hi")]
    }
    result = llm_call(state, config)

    invoked_messages = mock_llm.invoke.call_args[0][0]
    assert invoked_messages == state["messages"]
    assert result == {"messages": [AIMessage(content="Hello!")]}


def test_llm_call_does_not_inject_system_prompt():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Hello!")
    config = _make_config(mock_llm)

    state = {"messages": [HumanMessage(content="Hi")]}
    llm_call(state, config)

    invoked_messages = mock_llm.invoke.call_args[0][0]
    assert len(invoked_messages) == 1
    assert invoked_messages[0] == HumanMessage(content="Hi")


def test_llm_call_returns_tool_calls():
    mock_llm = MagicMock()
    ai_msg = AIMessage(
        content="Let me use the tool.",
        tool_calls=[{"name": "noop_tool", "args": {"input": "hello"}, "id": "1"}],
    )
    mock_llm.invoke.return_value = ai_msg
    config = _make_config(mock_llm)

    state = {"messages": [HumanMessage(content="Use noop")]}
    result = llm_call(state, config)

    assert result["messages"][0].tool_calls == [
        {"name": "noop_tool", "args": {"input": "hello"}, "id": "1", "type": "tool_call"}
    ]
