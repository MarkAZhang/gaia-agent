from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState

from agent.deps import AgentDeps


def llm_call(state: MessagesState, config: RunnableConfig) -> dict:
    deps: AgentDeps = config["configurable"]["deps"]
    response = deps.llm.invoke(state["messages"])
    return {"messages": [response]}
