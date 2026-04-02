from langfuse import Evaluation

from agent.agent_result import AgentResult


def latency_evaluator(*, output, **kwargs):
    """Evaluator that reports the task latency in seconds."""
    if not isinstance(output, AgentResult):
        return Evaluation(name="latency_seconds", value=0, comment="No AgentResult")

    return Evaluation(
        name="latency_seconds",
        value=output.latency_seconds,
        comment=f"Task completed in {output.latency_seconds:.2f}s",
    )
