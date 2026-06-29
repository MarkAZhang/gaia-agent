from evaluators.metrics.total_turns_evaluator import total_turns_evaluator


class TestTotalTurnsEvaluator:
    def test_returns_total_turns_from_output_metrics(self):
        outputs = {
            "answer": "answer",
            "metrics": {"total_turns": 7},
        }
        result = total_turns_evaluator(outputs=outputs)
        assert result["key"] == "total_turns"
        assert result["score"] == 7
        assert "7" in result["comment"]

    def test_returns_zero_for_missing_metrics(self):
        result = total_turns_evaluator(outputs={"answer": "plain string"})
        assert result["key"] == "total_turns"
        assert result["score"] == 0
