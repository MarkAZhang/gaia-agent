from dataclasses import dataclass
import time

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from agent.agent_response import AgentResponse, AgentRunMetrics
from agent.deps import AgentDeps
from langgraph.graph.state import CompiledStateGraph
from agent.graph import build_graph
from tools.web_search import create_web_search


def _compute_metrics(messages):
    """Extract token usage and turn counts from graph result messages."""
    input_tokens = 0
    output_tokens = 0
    total_turns = 0

    for msg in messages:
        if isinstance(msg, AIMessage):
            total_turns += 1
            usage = getattr(msg, "response_metadata", {}).get("usage", {})
            input_tokens += usage.get("input_tokens", 0)
            output_tokens += usage.get("output_tokens", 0)
        elif isinstance(msg, ToolMessage):
            total_turns += 1

    return input_tokens, output_tokens, total_turns


def _get_tools():
    return [create_web_search()]


@dataclass
class AgentCompiledGraphAndConfig:
    graph: CompiledStateGraph
    config: RunnableConfig


def build_agent_graph_and_config(langfuse_handler) -> AgentCompiledGraphAndConfig:
    tools = _get_tools()
    llm = ChatAnthropic(model="claude-opus-4-6").bind_tools(tools)
    config = RunnableConfig(
        configurable={"deps": AgentDeps(llm=llm)},
        callbacks=[langfuse_handler] if langfuse_handler else [],
    )
    return AgentCompiledGraphAndConfig(graph=build_graph(tools=tools), config=config)


def invoke_agent_with_user_message(input_str, langfuse_handler) -> AgentResponse:
    compiled_graph_and_config = build_agent_graph_and_config(langfuse_handler)

    start_time = time.monotonic()
    result = compiled_graph_and_config.graph.invoke(
        {"messages": [{"role": "user", "content": input_str}]},
        config=compiled_graph_and_config.config,
    )
    latency_seconds = time.monotonic() - start_time

    messages = result["messages"]
    final_answer = messages[-1].content if messages else "No answer found"
    input_tokens, output_tokens, total_turns = _compute_metrics(messages)

    return AgentResponse(
        answer=final_answer,
        metrics=AgentRunMetrics(
            latency_seconds=latency_seconds,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_turns=total_turns,
        ),
    )
