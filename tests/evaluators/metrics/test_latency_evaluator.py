from agent_graph.agent_response import AgentResponse, AgentRunMetrics
from evaluators.metrics.latency_evaluator import latency_evaluator


class TestLatencyEvaluator:
    def test_returns_latency_from_agent_response(self):
        agent_response = AgentResponse(
            answer="answer",
            deobfuscation_method="none",
            metrics=AgentRunMetrics(
                latency_seconds=5.23,
                input_tokens=0,
                output_tokens=0,
                total_turns=0,
            ),
        )
        result = latency_evaluator(output=agent_response)
        assert result.name == "latency_seconds"
        assert result.value == 5.23
        assert "5.23" in result.comment

    def test_returns_zero_for_non_agent_response(self):
        result = latency_evaluator(output="plain string")
        assert result.name == "latency_seconds"
        assert result.value == 0
        assert "No AgentResponse" in result.comment
