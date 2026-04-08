import re
from typing import Any

from langfuse import Evaluation

from agent.agent_response import AgentResponse


def gaia_score_evaluator(
    *, input: Any, output: Any, expected_output: Any, **kwargs: Any
) -> Evaluation:
    """
    Normalizes and compares the agent output with the GAIA ground truth.
    """
    answer = output.answer if isinstance(output, AgentResponse) else output

    def normalize(text: Any) -> str:
        if text == "" or text is None:
            return ""
        text = str(text).lower().strip()
        # Remove common units, currency symbols, and articles
        text = re.sub(r"[\$|%|kg|m|s|min|hr|days]", "", text)
        text = re.sub(r"\b(a|an|the)\b", "", text)
        # Standardize numbers (e.g., 15.0 -> 15)
        try:
            return str(float(text)).rstrip("0").rstrip(".")
        except ValueError:
            return text

    is_correct = normalize(answer) == normalize(expected_output)

    return Evaluation(
        name="exact_match",
        value=1.0 if is_correct else 0.0,
        comment=f"Expected: {expected_output} | Got: {answer}",
    )
