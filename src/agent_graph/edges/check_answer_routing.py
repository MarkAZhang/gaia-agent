from langgraph.graph import END

from agent_graph.context.agent_graph_state import AgentGraphState


def check_answer_routing(state: AgentGraphState) -> str:
    last_message = state["agent_messages"][-1]
    if last_message.type == "human":
        return "core_agent"
    return END
