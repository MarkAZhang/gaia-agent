from langchain_core.runnables import RunnableConfig

from agent_graph.agent_dependencies import AgentDependencies
from agent_graph.context.agent_graph_state import (
    AgentGraphState,
    get_messages_for_core_agent,
)


def core_agent(state: AgentGraphState, config: RunnableConfig) -> dict:
    deps: AgentDependencies = config["configurable"]["deps"]
    messages = get_messages_for_core_agent(state)
    response = deps.core_agent_model.invoke(messages)
    return {"agent_messages": [response]}
