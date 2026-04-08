from typing import Any

from langfuse import Evaluation

from agent.agent_response import AgentResponse


def total_turns_evaluator(*, output: Any, **kwargs: Any) -> Evaluation:
    """Evaluator that reports the total number of turns (LLM calls + tool calls)."""
    if not isinstance(output, AgentResponse):
        return Evaluation(name="total_turns", value=0, comment="No AgentResponse")

    return Evaluation(
        name="total_turns",
        value=output.metrics.total_turns,
        comment=f"Total turns: {output.metrics.total_turns}",
    )
