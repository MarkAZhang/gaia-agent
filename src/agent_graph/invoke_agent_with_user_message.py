import time
from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langfuse.langchain import CallbackHandler

from agent_graph.agent_response import AgentResponse, AgentRunMetrics
from agent_graph.build_agent_graph_and_config import build_agent_graph_and_config
from agent_graph.build_system_prompt import build_system_prompt
from agent_graph.guardrails.user_input_deobfuscator import deobfuscate_user_input


def _compute_metrics(messages: list[BaseMessage]) -> tuple[int, int, int]:
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


def invoke_agent_with_user_message(
    input_str: str,
    langfuse_handler: Optional[CallbackHandler],
    available_file_path: Optional[str] = None,
) -> AgentResponse:
    deobfuscation_result = deobfuscate_user_input(input_str)
    user_message = deobfuscation_result.text

    compiled_graph_and_config = build_agent_graph_and_config(langfuse_handler)
    system_prompt = build_system_prompt(available_file_path)

    start_time = time.monotonic()
    result = compiled_graph_and_config.graph.invoke(
        {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
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
        deobfuscation_method=deobfuscation_result.technique,
    )
