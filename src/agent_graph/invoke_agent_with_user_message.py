from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from agent_graph.agent_response import AgentResponse, AgentRunMetrics
from agent_graph.build_agent_graph_and_config import build_agent_graph_and_config
from agent_graph.build_system_prompt import build_system_prompt
from agent_graph.guardrails.user_input_deobfuscator import deobfuscate_user_input
from tools.shutdown_tools import shutdown_tools


def _compute_total_turns(messages: list[BaseMessage]) -> int:
    """Count LLM calls and tool calls from graph result messages."""
    total_turns = 0

    for msg in messages:
        if isinstance(msg, (AIMessage, ToolMessage)):
            total_turns += 1

    return total_turns


def invoke_agent_with_user_message(
    input_str: str,
    available_file_path: Optional[str] = None,
) -> AgentResponse:
    deobfuscation_result = deobfuscate_user_input(input_str)
    user_message = deobfuscation_result.text

    compiled_graph_and_config = build_agent_graph_and_config()
    system_prompt = build_system_prompt(available_file_path)

    try:
        result = compiled_graph_and_config.graph.invoke(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ]
            },
            config=compiled_graph_and_config.config,
        )
    finally:
        shutdown_tools()

    messages = result["messages"]
    final_answer = messages[-1].content if messages else "No answer found"
    total_turns = _compute_total_turns(messages)

    return AgentResponse(
        answer=final_answer,
        metrics=AgentRunMetrics(total_turns=total_turns),
        deobfuscation_method=deobfuscation_result.technique,
    )
