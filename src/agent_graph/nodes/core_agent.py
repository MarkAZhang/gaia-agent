from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState

from agent_graph.agent_dependencies import AgentDependencies


def core_agent(state: MessagesState, config: RunnableConfig) -> dict:
    deps: AgentDependencies = config["configurable"]["deps"]
    response = deps.core_agent_model.invoke(state["messages"])
    return {"messages": [response]}
