from langfuse import Evaluation

from agent.agent_result import AgentResult


def total_turns_evaluator(*, output, **kwargs):
    """Evaluator that reports the total number of turns (LLM calls + tool calls)."""
    if not isinstance(output, AgentResult):
        return Evaluation(name="total_turns", value=0, comment="No AgentResult")

    return Evaluation(
        name="total_turns",
        value=output.total_turns,
        comment=f"Total turns: {output.total_turns}",
    )
