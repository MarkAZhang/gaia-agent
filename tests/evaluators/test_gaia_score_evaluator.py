from agent.agent_response import AgentResponse, AgentRunMetrics
from evaluate_agent_on_dataset import DatasetItemInput
from evaluators.gaia_score_evaluator import gaia_score_evaluator


def _make_dataset_item_input() -> DatasetItemInput:
    return DatasetItemInput(task_id="t1", question="q", file_name="", file_path="")


def _make_agent_response(answer: str) -> AgentResponse:
    return AgentResponse(
        answer=answer,
        metrics=AgentRunMetrics(
            latency_seconds=1.0,
            input_tokens=10,
            output_tokens=5,
            total_turns=1,
        ),
    )


class TestGaiaScoreEvaluator:
    def test_exact_match(self):
        result = gaia_score_evaluator(
            input=_make_dataset_item_input(),
            output=_make_agent_response("42"),
            expected_output="42",
        )
        assert result.name == "exact_match"
        assert result.value == 1.0

    def test_no_match(self):
        result = gaia_score_evaluator(
            input=_make_dataset_item_input(),
            output=_make_agent_response("wrong"),
            expected_output="correct",
        )
        assert result.value == 0.0

    def test_handles_agent_response_output(self):
        agent_response = _make_agent_response("42")
        result = gaia_score_evaluator(
            input=_make_dataset_item_input(),
            output=agent_response,
            expected_output="42",
        )
        assert result.value == 1.0
        assert "42" in result.comment

    def test_agent_response_no_match(self):
        agent_response = _make_agent_response("wrong")
        result = gaia_score_evaluator(
            input=_make_dataset_item_input(),
            output=agent_response,
            expected_output="correct",
        )
        assert result.value == 0.0

    def test_normalizes_numeric_values(self):
        result = gaia_score_evaluator(
            input=_make_dataset_item_input(),
            output=_make_agent_response("15.0"),
            expected_output="15",
        )
        assert result.value == 1.0

    def test_case_insensitive(self):
        result = gaia_score_evaluator(
            input=_make_dataset_item_input(),
            output=_make_agent_response("HELLO"),
            expected_output="hello",
        )
        assert result.value == 1.0

    def test_empty_output(self):
        result = gaia_score_evaluator(
            input=_make_dataset_item_input(),
            output=_make_agent_response(""),
            expected_output="answer",
        )
        assert result.value == 0.0

    def test_returns_zero_when_output_is_not_agent_response(self):
        result = gaia_score_evaluator(
            input=_make_dataset_item_input(), output="42", expected_output="42"
        )
        assert result.value == 0.0
        assert result.comment == "No AgentResponse"
