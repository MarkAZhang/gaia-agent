from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

from agent.tools import noop_tool

TOOLS = [noop_tool]

SYSTEM_PROMPT = (
    "You are a ReAct agent. For each user request, follow this loop:\n"
    "1. THINK: Reason about the current state and what to do next.\n"
    "2. ACT: Call a tool if needed.\n"
    "3. OBSERVE: Review the tool result.\n"
    "Repeat until you can provide a final answer.\n"
    "Always explain your reasoning before acting."
)


def build_graph() -> StateGraph:
    llm = ChatAnthropic(model="claude-opus-4-6").bind_tools(TOOLS)

    def agent(state: MessagesState) -> dict:
        messages = state["messages"]
        if not messages or messages[0].type != "system":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        response = llm.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> str:
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return END

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, ["tools", END])
    graph.add_edge("tools", "agent")

    return graph.compile()
