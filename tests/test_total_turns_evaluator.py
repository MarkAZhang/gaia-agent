from agent.agent_result import AgentResult
from total_turns_evaluator import total_turns_evaluator


class TestTotalTurnsEvaluator:
    def test_returns_total_turns_from_agent_result(self):
        agent_result = AgentResult(
            answer="answer",
            latency_seconds=1.0,
            input_tokens=0,
            output_tokens=0,
            total_turns=7,
        )
        result = total_turns_evaluator(output=agent_result)
        assert result.name == "total_turns"
        assert result.value == 7
        assert "7" in result.comment

    def test_returns_zero_for_non_agent_result(self):
        result = total_turns_evaluator(output="plain string")
        assert result.name == "total_turns"
        assert result.value == 0
