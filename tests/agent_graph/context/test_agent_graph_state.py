from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from agent_graph.context.agent_graph_state import get_messages_for_core_agent


class TestGetMessagesForCoreAgent:
    def test_returns_agent_messages_when_no_tool_messages(self):
        state = {
            "agent_messages": [
                SystemMessage(content="sys"),
                HumanMessage(content="Hi"),
            ],
            "tool_messages": [],
        }
        result = get_messages_for_core_agent(state)
        assert result == state["agent_messages"]

    def test_appends_latest_tool_message(self):
        tool_msg_1 = ToolMessage(content="first result", tool_call_id="1")
        tool_msg_2 = ToolMessage(content="second result", tool_call_id="2")
        state = {
            "agent_messages": [HumanMessage(content="Hi")],
            "tool_messages": [tool_msg_1, tool_msg_2],
        }
        result = get_messages_for_core_agent(state)
        assert len(result) == 2
        assert result[0] == HumanMessage(content="Hi")
        assert result[1] == tool_msg_2

    def test_only_includes_last_tool_message(self):
        tool_msg_1 = ToolMessage(content="old result", tool_call_id="1")
        tool_msg_2 = ToolMessage(content="new result", tool_call_id="2")
        state = {
            "agent_messages": [
                SystemMessage(content="sys"),
                HumanMessage(content="query"),
            ],
            "tool_messages": [tool_msg_1, tool_msg_2],
        }
        result = get_messages_for_core_agent(state)
        assert len(result) == 3
        assert result[-1] == tool_msg_2
        assert tool_msg_1 not in result

    def test_preserves_agent_message_order(self):
        sys_msg = SystemMessage(content="sys")
        human_msg = HumanMessage(content="query")
        state = {
            "agent_messages": [sys_msg, human_msg],
            "tool_messages": [],
        }
        result = get_messages_for_core_agent(state)
        assert result[0] == sys_msg
        assert result[1] == human_msg
