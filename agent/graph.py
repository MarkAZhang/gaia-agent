from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

from agent.deps import AgentDeps
from agent.nodes.llm_call import llm_call
from agent.nodes.should_continue import should_continue
from tools.noop_tool import noop_tool

TOOLS = [noop_tool]


def build_graph(config: RunnableConfig | None = None):
    if config is None:
        llm = ChatAnthropic(model="claude-opus-4-6").bind_tools(TOOLS)
        config = RunnableConfig(configurable={"deps": AgentDeps(llm=llm)})

    graph = StateGraph(MessagesState)
    graph.add_node("llm_call", llm_call)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "llm_call")
    graph.add_conditional_edges("llm_call", should_continue, ["tools", END])
    graph.add_edge("tools", "llm_call")

    return graph.compile(), config
