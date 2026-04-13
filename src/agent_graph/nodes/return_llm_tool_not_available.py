from agent_graph.context.agent_graph_state import AgentGraphState


def return_llm_tool_not_available(state: AgentGraphState) -> dict:
    last_message = state["agent_messages"][-1]
    content = last_message.content if isinstance(last_message.content, str) else ""
    last_line = content.strip().splitlines()[-1] if content.strip() else ""
    return {"agent_messages": [{"role": "ai", "content": last_line}]}
