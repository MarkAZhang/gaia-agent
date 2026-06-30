from typing import Any


def latency_evaluator(
    *, outputs: dict[str, Any], **kwargs: Any
) -> dict[str, Any]:
    """Evaluator that reports the task latency in seconds."""
    metrics = outputs.get("metrics")
    if not isinstance(metrics, dict):
        return {
            "key": "latency_seconds",
            "score": 0,
            "comment": "No metrics in output",
        }

    latency_seconds = metrics.get("latency_seconds", 0)
    return {
        "key": "latency_seconds",
        "score": latency_seconds,
        "comment": f"Task completed in {latency_seconds:.2f}s",
    }
