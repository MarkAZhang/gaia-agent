from langfuse import Evaluation

from agent.agent_result import AgentResult


def input_tokens_evaluator(*, output, **kwargs):
    """Evaluator that reports the total input tokens used."""
    if not isinstance(output, AgentResult):
        return Evaluation(name="input_tokens", value=0, comment="No AgentResult")

    return Evaluation(
        name="input_tokens",
        value=output.input_tokens,
        comment=f"Total input tokens: {output.input_tokens}",
    )


def output_tokens_evaluator(*, output, **kwargs):
    """Evaluator that reports the total output tokens used."""
    if not isinstance(output, AgentResult):
        return Evaluation(name="output_tokens", value=0, comment="No AgentResult")

    return Evaluation(
        name="output_tokens",
        value=output.output_tokens,
        comment=f"Total output tokens: {output.output_tokens}",
    )
