from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState

from agent.deps import AgentDeps
from agent.prompts.get_prompt import get_prompt

SYSTEM_PROMPT = get_prompt("react_system_prompt")


def llm_call(state: MessagesState, config: RunnableConfig) -> dict:
    deps: AgentDeps = config["configurable"]["deps"]
    messages = state["messages"]
    if not messages or messages[0].type != "system":
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    response = deps.llm.invoke(messages)
    return {"messages": [response]}
