from typing import Any


def total_turns_evaluator(
    *, outputs: dict[str, Any], **kwargs: Any
) -> dict[str, Any]:
    """Evaluator that reports the total number of turns (LLM calls + tool calls)."""
    metrics = outputs.get("metrics")
    if not isinstance(metrics, dict):
        return {
            "key": "total_turns",
            "score": 0,
            "comment": "No metrics in output",
        }

    total_turns = metrics.get("total_turns", 0)
    return {
        "key": "total_turns",
        "score": total_turns,
        "comment": f"Total turns: {total_turns}",
    }
