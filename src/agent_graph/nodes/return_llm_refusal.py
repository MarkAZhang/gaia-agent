from agent_graph.context.agent_graph_state import AgentGraphState


def return_llm_refusal(state: AgentGraphState) -> dict:
    return {"agent_messages": [{"role": "ai", "content": "LLM refused to continue"}]}
