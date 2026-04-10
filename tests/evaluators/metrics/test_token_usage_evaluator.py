from agent_graph.agent_response import AgentResponse, AgentRunMetrics
from evaluators.metrics.token_usage_evaluator import (
    input_tokens_evaluator,
    output_tokens_evaluator,
)


class TestInputTokensEvaluator:
    def test_returns_input_tokens_from_agent_response(self):
        agent_response = AgentResponse(
            answer="answer",
            metrics=AgentRunMetrics(
                latency_seconds=1.0,
                input_tokens=500,
                output_tokens=100,
                total_turns=2,
            ),
        )
        result = input_tokens_evaluator(output=agent_response)
        assert result.name == "input_tokens"
        assert result.value == 500
        assert "500" in result.comment

    def test_returns_zero_for_non_agent_response(self):
        result = input_tokens_evaluator(output="plain string")
        assert result.name == "input_tokens"
        assert result.value == 0


class TestOutputTokensEvaluator:
    def test_returns_output_tokens_from_agent_response(self):
        agent_response = AgentResponse(
            answer="answer",
            metrics=AgentRunMetrics(
                latency_seconds=1.0,
                input_tokens=500,
                output_tokens=150,
                total_turns=2,
            ),
        )
        result = output_tokens_evaluator(output=agent_response)
        assert result.name == "output_tokens"
        assert result.value == 150
        assert "150" in result.comment

    def test_returns_zero_for_non_agent_response(self):
        result = output_tokens_evaluator(output="plain string")
        assert result.name == "output_tokens"
        assert result.value == 0
