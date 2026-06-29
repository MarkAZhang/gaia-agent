from evaluators.metrics.token_usage_evaluator import (
    input_tokens_evaluator,
    output_tokens_evaluator,
)


class TestInputTokensEvaluator:
    def test_returns_input_tokens_from_output_metrics(self):
        outputs = {
            "answer": "answer",
            "metrics": {"input_tokens": 500, "output_tokens": 100},
        }
        result = input_tokens_evaluator(outputs=outputs)
        assert result["key"] == "input_tokens"
        assert result["score"] == 500
        assert "500" in result["comment"]

    def test_returns_zero_for_missing_metrics(self):
        result = input_tokens_evaluator(outputs={"answer": "plain string"})
        assert result["key"] == "input_tokens"
        assert result["score"] == 0


class TestOutputTokensEvaluator:
    def test_returns_output_tokens_from_output_metrics(self):
        outputs = {
            "answer": "answer",
            "metrics": {"input_tokens": 500, "output_tokens": 150},
        }
        result = output_tokens_evaluator(outputs=outputs)
        assert result["key"] == "output_tokens"
        assert result["score"] == 150
        assert "150" in result["comment"]

    def test_returns_zero_for_missing_metrics(self):
        result = output_tokens_evaluator(outputs={"answer": "plain string"})
        assert result["key"] == "output_tokens"
        assert result["score"] == 0
