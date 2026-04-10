import os
from dataclasses import dataclass
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langfuse.langchain import CallbackHandler
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from agent_graph.agent_dependencies import AgentDependencies
from agent_graph.edges.check_answer_routing import check_answer_routing
from agent_graph.edges.should_continue import should_continue
from agent_graph.nodes.check_and_get_final_answer import check_and_get_final_answer
from agent_graph.nodes.core_agent import core_agent
from agent_graph.nodes.return_llm_refusal import return_llm_refusal
from agent_graph.nodes.return_llm_tool_not_available import (
    return_llm_tool_not_available,
)
from tools.code_runner import execute_code_file, execute_code_snippet
from tools.document_parser import parse_document
from llm_wrappers.gemini_image_analyzer import GeminiImageAnalyzer
from tools.image_analyzer import create_image_analyzer_tool
from tools.web_searcher import create_web_search

IMAGE_ANALYZER_MODEL = "gemini-3.1-pro-preview"


def _get_tools() -> list[BaseTool]:
    return [
        create_web_search(),
        execute_code_snippet,
        execute_code_file,
        parse_document,
        create_image_analyzer_tool(
            analyzer=GeminiImageAnalyzer(
                model=IMAGE_ANALYZER_MODEL,
                api_key=os.environ["GEMINI_API_KEY"],
            ),
        ),
    ]


def _build_graph(tools: list[BaseTool]) -> CompiledStateGraph:
    graph = StateGraph(MessagesState)
    graph.add_node("core_agent", core_agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("check_and_get_final_answer", check_and_get_final_answer)
    graph.add_node("return_llm_refusal", return_llm_refusal)
    graph.add_node("return_llm_tool_not_available", return_llm_tool_not_available)

    graph.add_edge(START, "core_agent")
    graph.add_conditional_edges(
        "core_agent",
        should_continue,
        [
            "tools",
            "check_and_get_final_answer",
            "return_llm_refusal",
            "return_llm_tool_not_available",
        ],
    )
    graph.add_edge("tools", "core_agent")
    graph.add_edge("return_llm_refusal", END)
    graph.add_edge("return_llm_tool_not_available", END)
    graph.add_conditional_edges(
        "check_and_get_final_answer",
        check_answer_routing,
        ["core_agent", END],
    )

    return graph.compile()


@dataclass
class AgentCompiledGraphAndConfig:
    graph: CompiledStateGraph
    config: RunnableConfig


def build_agent_graph_and_config(
    langfuse_handler: Optional[CallbackHandler],
) -> AgentCompiledGraphAndConfig:
    tools = _get_tools()
    llm = ChatAnthropic(model="claude-opus-4-6").bind_tools(tools)
    config = RunnableConfig(
        configurable={
            "deps": AgentDependencies(
                core_agent_model=llm,
            )
        },
        callbacks=[langfuse_handler] if langfuse_handler else [],
    )
    return AgentCompiledGraphAndConfig(graph=_build_graph(tools=tools), config=config)
