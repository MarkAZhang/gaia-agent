from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode

from agent_graph.context.agent_graph_state import AgentGraphState


def tool_node(tools: list[BaseTool]):
    prebuilt_tool_node = ToolNode(tools)

    def run_tools(state: AgentGraphState) -> dict:
        # ToolNode expects {"messages": [<ai_message_with_tool_calls>]}.
        # The last agent message contains the tool calls.
        result = prebuilt_tool_node.invoke(
            {"messages": [state["agent_messages"][-1]]}
        )
        return {"tool_messages": result["messages"]}

    return run_tools
