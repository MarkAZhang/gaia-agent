from agent_graph.context.agent_graph_state import AgentGraphState


def should_continue(state: AgentGraphState) -> str:
    last_message = state["agent_messages"][-1]
    response_metadata = last_message.response_metadata or {}
    if response_metadata.get("stop_reason") == "refusal":
        return "return_llm_refusal"
    if last_message.tool_calls:
        return "tools"
    content = last_message.content if isinstance(last_message.content, str) else ""
    last_line = content.strip().splitlines()[-1] if content.strip() else ""
    if last_line.startswith("Tool not available:"):
        return "return_llm_tool_not_available"
    return "check_and_get_final_answer"
