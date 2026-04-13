import warnings

import pytest
from agent_graph.agent_response import AgentResponse, AgentRunMetrics
from evaluate_agent_on_dataset import DatasetItemInput
from evaluators.gaia_score_evaluator import (
    gaia_score_evaluator,
    normalize_number_str,
    normalize_str,
    question_scorer,
    split_string,
)


def _make_dataset_item_input() -> DatasetItemInput:
    return DatasetItemInput(task_id="t1", question="q", file_name="", file_path="")


def _make_agent_response(answer: str) -> AgentResponse:
    return AgentResponse(
        answer=answer,
        deobfuscation_method="none",
        metrics=AgentRunMetrics(
            latency_seconds=1.0,
            input_tokens=10,
            output_tokens=5,
            total_turns=1,
        ),
    )


class TestGaiaScoreEvaluator:
    def test_returns_zero_when_output_is_not_agent_response(self):
        result = gaia_score_evaluator(
            input=_make_dataset_item_input(), output="42", expected_output="42"
        )
        assert result.value == 0.0
        assert result.comment == "No AgentResponse"

    def test_returns_zero_for_dict_output(self):
        result = gaia_score_evaluator(
            input=_make_dataset_item_input(),
            output={"answer": "42"},
            expected_output="42",
        )
        assert result.value == 0.0
        assert result.comment == "No AgentResponse"

    def test_exact_match_number(self):
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

    def test_comment_includes_expected_and_actual(self):
        result = gaia_score_evaluator(
            input=_make_dataset_item_input(),
            output=_make_agent_response("foo"),
            expected_output="bar",
        )
        assert "bar" in result.comment
        assert "foo" in result.comment

    def test_string_match_case_insensitive(self):
        result = gaia_score_evaluator(
            input=_make_dataset_item_input(),
            output=_make_agent_response("HELLO"),
            expected_output="hello",
        )
        assert result.value == 1.0

    def test_empty_answer_does_not_match(self):
        result = gaia_score_evaluator(
            input=_make_dataset_item_input(),
            output=_make_agent_response(""),
            expected_output="answer",
        )
        assert result.value == 0.0


class TestNormalizeNumberStr:
    def test_plain_integer(self):
        assert normalize_number_str("42") == 42.0

    def test_plain_float(self):
        assert normalize_number_str("3.14") == 3.14

    def test_strips_dollar_sign(self):
        assert normalize_number_str("$100") == 100.0

    def test_strips_percent_sign(self):
        assert normalize_number_str("50%") == 50.0

    def test_strips_commas(self):
        assert normalize_number_str("1,000,000") == 1_000_000.0

    def test_strips_multiple_symbols(self):
        assert normalize_number_str("$1,234.56") == 1234.56

    def test_non_numeric_returns_inf(self):
        assert normalize_number_str("abc") == float("inf")

    def test_negative_number(self):
        assert normalize_number_str("-7.5") == -7.5


class TestSplitString:
    def test_split_on_comma(self):
        assert split_string("a,b,c") == ["a", "b", "c"]

    def test_split_on_semicolon(self):
        assert split_string("x;y;z") == ["x", "y", "z"]

    def test_mixed_delimiters(self):
        assert split_string("a,b;c") == ["a", "b", "c"]

    def test_no_delimiter(self):
        assert split_string("hello") == ["hello"]

    def test_custom_char_list(self):
        assert split_string("a|b|c", char_list=["|"]) == ["a", "b", "c"]


class TestNormalizeStr:
    def test_lowercases(self):
        assert normalize_str("HELLO") == "hello"

    def test_removes_whitespace(self):
        assert normalize_str("sea gull") == "seagull"

    def test_removes_punctuation_by_default(self):
        assert normalize_str("hello, world!") == "helloworld"

    def test_keeps_punctuation_when_specified(self):
        assert normalize_str("hello, world!", remove_punct=False) == "hello,world!"

    def test_tabs_and_newlines_removed(self):
        assert normalize_str("a\tb\nc") == "abc"


class TestQuestionScorer:
    # --- numeric ground truth ---
    def test_numeric_exact(self):
        assert question_scorer("42", "42") is True

    def test_numeric_float_equivalent(self):
        assert question_scorer("15.0", "15") is True

    def test_numeric_with_currency(self):
        assert question_scorer("$1,234", "1234") is True

    def test_numeric_mismatch(self):
        assert question_scorer("99", "42") is False

    def test_numeric_answer_non_numeric(self):
        assert question_scorer("abc", "42") is False

    # --- string ground truth ---
    def test_string_exact(self):
        assert question_scorer("hello", "hello") is True

    def test_string_case_insensitive(self):
        assert question_scorer("HELLO", "hello") is True

    def test_string_ignores_whitespace(self):
        assert question_scorer("sea gull", "seagull") is True

    def test_string_ignores_punctuation(self):
        assert question_scorer("hello!", "hello") is True

    def test_string_mismatch(self):
        assert question_scorer("foo", "bar") is False

    # --- list ground truth (comma/semicolon) ---
    def test_list_exact_match(self):
        assert question_scorer("a,b,c", "a,b,c") is True

    def test_list_with_spaces(self):
        assert question_scorer("a, b, c", "a, b, c") is True

    def test_list_case_insensitive(self):
        assert question_scorer("A,B,C", "a,b,c") is True

    def test_list_semicolon(self):
        assert question_scorer("x;y;z", "x;y;z") is True

    def test_list_mismatch_element(self):
        assert question_scorer("a,b,d", "a,b,c") is False

    def test_list_different_length_returns_false(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            assert question_scorer("a,b", "a,b,c") is False

    def test_list_different_length_warns(self):
        with pytest.warns(UserWarning, match="different lengths"):
            question_scorer("a,b", "a,b,c")

    def test_list_with_numeric_elements(self):
        assert question_scorer("1,2,3", "1,2,3") is True

    def test_list_with_mixed_types(self):
        assert question_scorer("hello,42", "hello,42") is True

    def test_list_numeric_element_with_currency(self):
        assert question_scorer("$100,hello", "100,hello") is True

    # --- None handling ---
    def test_none_answer_treated_as_string(self):
        assert question_scorer(None, "hello") is False  # type: ignore[arg-type]

    def test_none_answer_vs_none_string(self):
        assert question_scorer(None, "None") is True  # type: ignore[arg-type]
