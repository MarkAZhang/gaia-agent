from langchain_core.messages import AIMessage

from agent_graph.nodes.return_llm_tool_not_available import return_llm_tool_not_available


def test_returns_tool_not_available_last_line():
    msg = AIMessage(
        content="I looked for a tool.\nTool not available: No calculator tool found"
    )
    state = {"messages": [msg]}
    result = return_llm_tool_not_available(state)
    assert result == {
        "messages": [
            {
                "role": "ai",
                "content": "Tool not available: No calculator tool found",
            }
        ]
    }


def test_returns_tool_not_available_single_line():
    msg = AIMessage(content="Tool not available: No matching tool")
    state = {"messages": [msg]}
    result = return_llm_tool_not_available(state)
    assert result == {
        "messages": [{"role": "ai", "content": "Tool not available: No matching tool"}]
    }


def test_returns_empty_string_for_empty_content():
    msg = AIMessage(content="")
    state = {"messages": [msg]}
    result = return_llm_tool_not_available(state)
    assert result == {"messages": [{"role": "ai", "content": ""}]}
