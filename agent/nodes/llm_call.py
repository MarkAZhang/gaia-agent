from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState

from agent.deps import AgentDeps

SYSTEM_PROMPT = (
    "You are a ReAct agent. For each user request, follow this loop:\n"
    "1. THINK: Reason about the current state and what to do next.\n"
    "2. ACT: Call a tool if needed.\n"
    "3. OBSERVE: Review the tool result.\n"
    "Repeat until you can provide a final answer.\n"
    "Always explain your reasoning before acting."
)


def llm_call(state: MessagesState, config: RunnableConfig) -> dict:
    deps: AgentDeps = config["configurable"]["deps"]
    messages = state["messages"]
    if not messages or messages[0].type != "system":
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    response = deps.llm.invoke(messages)
    return {"messages": [response]}
