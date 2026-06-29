from typing import Any


def input_tokens_evaluator(
    *, outputs: dict[str, Any], **kwargs: Any
) -> dict[str, Any]:
    """Evaluator that reports the total input tokens used."""
    metrics = outputs.get("metrics")
    if not isinstance(metrics, dict):
        return {
            "key": "input_tokens",
            "score": 0,
            "comment": "No metrics in output",
        }

    input_tokens = metrics.get("input_tokens", 0)
    return {
        "key": "input_tokens",
        "score": input_tokens,
        "comment": f"Total input tokens: {input_tokens}",
    }


def output_tokens_evaluator(
    *, outputs: dict[str, Any], **kwargs: Any
) -> dict[str, Any]:
    """Evaluator that reports the total output tokens used."""
    metrics = outputs.get("metrics")
    if not isinstance(metrics, dict):
        return {
            "key": "output_tokens",
            "score": 0,
            "comment": "No metrics in output",
        }

    output_tokens = metrics.get("output_tokens", 0)
    return {
        "key": "output_tokens",
        "score": output_tokens,
        "comment": f"Total output tokens: {output_tokens}",
    }
