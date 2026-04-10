from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, Union

from langfuse import Evaluation

from agent_graph.agent_response import AgentResponse

if TYPE_CHECKING:
    from evaluate_agent_on_dataset import DatasetItemInput


def gaia_score_evaluator(
    *,
    input: DatasetItemInput,
    output: Union[str, Dict[str, Any], AgentResponse],
    expected_output: str,
    **kwargs: Any,
) -> Evaluation:
    """
    Normalizes and compares the agent output with the GAIA ground truth.
    """
    if not isinstance(output, AgentResponse):
        return Evaluation(name="exact_match", value=0.0, comment="No AgentResponse")

    answer = output.answer

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
