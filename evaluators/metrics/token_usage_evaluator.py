from typing import Any, Dict, Union

from langfuse import Evaluation

from agent.agent_response import AgentResponse


def input_tokens_evaluator(
    *, output: Union[str, Dict[str, Any], AgentResponse], **kwargs: Any
) -> Evaluation:
    """Evaluator that reports the total input tokens used."""
    if not isinstance(output, AgentResponse):
        return Evaluation(name="input_tokens", value=0, comment="No AgentResponse")

    return Evaluation(
        name="input_tokens",
        value=output.metrics.input_tokens,
        comment=f"Total input tokens: {output.metrics.input_tokens}",
    )


def output_tokens_evaluator(
    *, output: Union[str, Dict[str, Any], AgentResponse], **kwargs: Any
) -> Evaluation:
    """Evaluator that reports the total output tokens used."""
    if not isinstance(output, AgentResponse):
        return Evaluation(name="output_tokens", value=0, comment="No AgentResponse")

    return Evaluation(
        name="output_tokens",
        value=output.metrics.output_tokens,
        comment=f"Total output tokens: {output.metrics.output_tokens}",
    )
