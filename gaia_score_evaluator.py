import re
from langfuse import Evaluation


def gaia_score_evaluator(*, input, output, expected_output, **kwargs):
    """
    Normalizes and compares the agent output with the GAIA ground truth.
    """

    def normalize(text):
        if not text:
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

    is_correct = normalize(output) == normalize(expected_output)

    return Evaluation(
        name="exact_match",
        value=1.0 if is_correct else 0.0,
        comment=f"Expected: {expected_output} | Got: {output}",
    )
