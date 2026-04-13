from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent_graph.nodes.memory_management import memory_management


def test_removes_tool_messages_before_last_ai_message():
    messages = [
        HumanMessage(content="Search for X"),
        AIMessage(
            content="Searching.",
            tool_calls=[{"name": "search", "args": {}, "id": "call_1"}],
        ),
        ToolMessage(content="Long result from search", tool_call_id="call_1", id="t1"),
        AIMessage(
            content="Let me search again.",
            tool_calls=[{"name": "search", "args": {}, "id": "call_2"}],
        ),
        ToolMessage(
            content="Another long result", tool_call_id="call_2", id="t2"
        ),
    ]
    state = {"messages": messages}
    result = memory_management(state)

    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "removed"
    assert result["messages"][0].tool_call_id == "call_1"
    assert result["messages"][0].id == "t1"


def test_keeps_tool_messages_after_last_ai_message():
    messages = [
        HumanMessage(content="Search for X"),
        AIMessage(
            content="Searching.",
            tool_calls=[{"name": "search", "args": {}, "id": "call_1"}],
        ),
        ToolMessage(content="Result from search", tool_call_id="call_1", id="t1"),
    ]
    state = {"messages": messages}
    result = memory_management(state)

    assert len(result["messages"]) == 0


def test_no_messages_when_no_ai_message():
    messages = [
        HumanMessage(content="Hello"),
    ]
    state = {"messages": messages}
    result = memory_management(state)

    assert len(result["messages"]) == 0


def test_removes_multiple_old_tool_messages():
    messages = [
        HumanMessage(content="Search"),
        AIMessage(
            content="Searching.",
            tool_calls=[
                {"name": "search", "args": {}, "id": "call_1"},
                {"name": "search", "args": {}, "id": "call_2"},
            ],
        ),
        ToolMessage(content="Result 1", tool_call_id="call_1", id="t1"),
        ToolMessage(content="Result 2", tool_call_id="call_2", id="t2"),
        AIMessage(
            content="Need more info.",
            tool_calls=[{"name": "search", "args": {}, "id": "call_3"}],
        ),
        ToolMessage(content="Result 3", tool_call_id="call_3", id="t3"),
    ]
    state = {"messages": messages}
    result = memory_management(state)

    assert len(result["messages"]) == 2
    assert all(m.content == "removed" for m in result["messages"])
    assert result["messages"][0].tool_call_id == "call_1"
    assert result["messages"][1].tool_call_id == "call_2"


def test_preserves_tool_call_id_on_removed_messages():
    messages = [
        HumanMessage(content="Search"),
        AIMessage(
            content="First search.",
            tool_calls=[{"name": "search", "args": {}, "id": "call_1"}],
        ),
        ToolMessage(
            content="Very long result text", tool_call_id="call_1", id="t1"
        ),
        AIMessage(content="Ans: 42"),
    ]
    state = {"messages": messages}
    result = memory_management(state)

    assert len(result["messages"]) == 1
    removed = result["messages"][0]
    assert removed.content == "removed"
    assert removed.tool_call_id == "call_1"
    assert removed.id == "t1"


def test_empty_messages():
    state = {"messages": []}
    result = memory_management(state)

    assert len(result["messages"]) == 0


def test_multiple_rounds_of_tool_calls_removes_all_old():
    messages = [
        HumanMessage(content="Question"),
        AIMessage(
            content="Round 1.",
            tool_calls=[{"name": "t", "args": {}, "id": "c1"}],
        ),
        ToolMessage(content="R1", tool_call_id="c1", id="t1"),
        AIMessage(
            content="Round 2.",
            tool_calls=[{"name": "t", "args": {}, "id": "c2"}],
        ),
        ToolMessage(content="R2", tool_call_id="c2", id="t2"),
        AIMessage(
            content="Round 3.",
            tool_calls=[{"name": "t", "args": {}, "id": "c3"}],
        ),
        ToolMessage(content="R3", tool_call_id="c3", id="t3"),
    ]
    state = {"messages": messages}
    result = memory_management(state)

    assert len(result["messages"]) == 2
    assert result["messages"][0].tool_call_id == "c1"
    assert result["messages"][1].tool_call_id == "c2"


def test_already_removed_tool_messages_stay_removed():
    messages = [
        HumanMessage(content="Search"),
        AIMessage(
            content="First.",
            tool_calls=[{"name": "t", "args": {}, "id": "c1"}],
        ),
        ToolMessage(content="removed", tool_call_id="c1", id="t1"),
        AIMessage(
            content="Second.",
            tool_calls=[{"name": "t", "args": {}, "id": "c2"}],
        ),
        ToolMessage(content="New result", tool_call_id="c2", id="t2"),
    ]
    state = {"messages": messages}
    result = memory_management(state)

    # Already-removed tool messages should not be replaced again
    assert len(result["messages"]) == 0
