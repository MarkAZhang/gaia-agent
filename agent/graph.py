from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

from agent.edges.check_answer_routing import check_answer_routing
from agent.edges.should_continue import should_continue
from agent.nodes.check_and_get_final_answer import check_and_get_final_answer
from agent.nodes.llm_call import llm_call
from agent.nodes.return_llm_refusal import return_llm_refusal
from agent.nodes.return_llm_tool_not_available import return_llm_tool_not_available
from tools.web_search import create_web_search


def build_graph(tools=None):
    if tools is None:
        tools = [create_web_search()]

    graph = StateGraph(MessagesState)
    graph.add_node("llm_call", llm_call)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("check_and_get_final_answer", check_and_get_final_answer)
    graph.add_node("return_llm_refusal", return_llm_refusal)
    graph.add_node("return_llm_tool_not_available", return_llm_tool_not_available)

    graph.add_edge(START, "llm_call")
    graph.add_conditional_edges(
        "llm_call",
        should_continue,
        [
            "tools",
            "check_and_get_final_answer",
            "return_llm_refusal",
            "return_llm_tool_not_available",
        ],
    )
    graph.add_edge("tools", "llm_call")
    graph.add_edge("return_llm_refusal", END)
    graph.add_edge("return_llm_tool_not_available", END)
    graph.add_conditional_edges(
        "check_and_get_final_answer",
        check_answer_routing,
        ["llm_call", END],
    )

    return graph.compile()
