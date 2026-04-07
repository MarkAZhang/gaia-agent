from dataclasses import dataclass
import time
from typing import Iterable, Optional, Sequence, Union

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from agent.agent_response import AgentResponse, AgentRunMetrics
from agent.deps import AgentDeps
from agent.prompts.get_prompt import get_prompt
from langgraph.graph.state import CompiledStateGraph
from agent.graph import build_graph
from tools.web_search import create_web_search


def _build_file_paths_block(available_file_paths: Optional[Sequence[str]]) -> str:
    """Render the file-paths section injected into the system prompt.

    Returns an empty string when no paths were provided so the conclusion
    section renders cleanly.
    """
    if not available_file_paths:
        return ""
    lines = "\n".join(f"- {p}" for p in available_file_paths)
    return f"Available file paths:\n{lines}\n\n"


def build_system_prompt(
    available_file_paths: Optional[Sequence[str]] = None,
) -> str:
    return get_prompt(
        "react_system_prompt",
        file_paths_block=_build_file_paths_block(available_file_paths),
    )


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


def invoke_agent_with_user_message(
    input_str,
    langfuse_handler,
    available_file_paths: Optional[Union[str, Iterable[str]]] = None,
) -> AgentResponse:
    if isinstance(available_file_paths, str):
        file_paths: Optional[Sequence[str]] = [available_file_paths]
    elif available_file_paths is None:
        file_paths = None
    else:
        file_paths = list(available_file_paths)

    compiled_graph_and_config = build_agent_graph_and_config(langfuse_handler)
    system_prompt = build_system_prompt(file_paths)

    start_time = time.monotonic()
    result = compiled_graph_and_config.graph.invoke(
        {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_str},
            ]
        },
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
