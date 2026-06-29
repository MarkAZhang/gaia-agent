from evaluators.metrics.latency_evaluator import latency_evaluator


class TestLatencyEvaluator:
    def test_returns_latency_from_output_metrics(self):
        outputs = {
            "answer": "answer",
            "metrics": {"latency_seconds": 5.23},
        }
        result = latency_evaluator(outputs=outputs)
        assert result["key"] == "latency_seconds"
        assert result["score"] == 5.23
        assert "5.23" in result["comment"]

    def test_returns_zero_for_missing_metrics(self):
        result = latency_evaluator(outputs={"answer": "plain string"})
        assert result["key"] == "latency_seconds"
        assert result["score"] == 0
        assert "No metrics in output" in result["comment"]
