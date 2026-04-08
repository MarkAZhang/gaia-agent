from typing import Any, Dict, Union

from langfuse import Evaluation

from agent.agent_response import AgentResponse


def latency_evaluator(
    *, output: Union[str, Dict[str, Any], AgentResponse], **kwargs: Any
) -> Evaluation:
    """Evaluator that reports the task latency in seconds."""
    if not isinstance(output, AgentResponse):
        return Evaluation(name="latency_seconds", value=0, comment="No AgentResponse")

    return Evaluation(
        name="latency_seconds",
        value=output.metrics.latency_seconds,
        comment=f"Task completed in {output.metrics.latency_seconds:.2f}s",
    )
