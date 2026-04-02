from agent.agent_response import AgentResponse, AgentRunMetrics
from evaluators.metrics.total_turns_evaluator import total_turns_evaluator


class TestTotalTurnsEvaluator:
    def test_returns_total_turns_from_agent_response(self):
        agent_response = AgentResponse(
            answer="answer",
            metrics=AgentRunMetrics(
                latency_seconds=1.0,
                input_tokens=0,
                output_tokens=0,
                total_turns=7,
            ),
        )
        result = total_turns_evaluator(output=agent_response)
        assert result.name == "total_turns"
        assert result.value == 7
        assert "7" in result.comment

    def test_returns_zero_for_non_agent_response(self):
        result = total_turns_evaluator(output="plain string")
        assert result.name == "total_turns"
        assert result.value == 0
