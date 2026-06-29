from unittest.mock import MagicMock, patch

from agent_graph.agent_response import AgentResponse, AgentRunMetrics
from evaluate_agent_on_dataset import evaluate_agent_on_dataset, run_agent_for_dataset_item


def _make_agent_response(**overrides):
    defaults = {
        "answer": "answer",
        "metrics": AgentRunMetrics(
            latency_seconds=1.0,
            input_tokens=0,
            output_tokens=0,
            total_turns=0,
        ),
        "deobfuscation_method": "none",
    }
    defaults.update(overrides)
    return AgentResponse(**defaults)


class TestRunAgentForDatasetItem:
    @patch("evaluate_agent_on_dataset.invoke_agent_with_user_message")
    @patch("evaluate_agent_on_dataset.create_langsmith_tracer")
    def test_passes_question_to_invoke_agent(
        self, mock_create_tracer, mock_invoke
    ):
        mock_invoke.return_value = _make_agent_response(
            metrics=AgentRunMetrics(
                latency_seconds=1.0,
                input_tokens=10,
                output_tokens=5,
                total_turns=1,
            ),
        )
        mock_tracer = MagicMock()
        mock_create_tracer.return_value = mock_tracer

        inputs = {
            "task_id": "task1",
            "question": "What is 2+2?",
            "file_name": "",
            "file_path": "",
        }

        run_agent_for_dataset_item(inputs)

        mock_invoke.assert_called_once_with(
            "What is 2+2?",
            tracing_handler=mock_tracer,
            available_file_path=None,
        )

    @patch("evaluate_agent_on_dataset.invoke_agent_with_user_message")
    @patch("evaluate_agent_on_dataset.create_langsmith_tracer")
    def test_returns_serialized_output(self, mock_create_tracer, mock_invoke):
        mock_invoke.return_value = _make_agent_response(
            answer="the answer is 4",
            metrics=AgentRunMetrics(
                latency_seconds=2.5,
                input_tokens=100,
                output_tokens=50,
                total_turns=3,
            ),
        )
        mock_create_tracer.return_value = MagicMock()

        inputs = {
            "task_id": "task1",
            "question": "What is 2+2?",
            "file_name": "",
            "file_path": "",
        }

        result = run_agent_for_dataset_item(inputs)

        assert result["answer"] == "the answer is 4"
        assert result["metrics"]["latency_seconds"] == 2.5
        assert result["metrics"]["input_tokens"] == 100
        assert result["metrics"]["output_tokens"] == 50
        assert result["metrics"]["total_turns"] == 3

    @patch("evaluate_agent_on_dataset.invoke_agent_with_user_message")
    @patch("evaluate_agent_on_dataset.create_langsmith_tracer")
    def test_creates_langsmith_tracer(self, mock_create_tracer, mock_invoke):
        mock_invoke.return_value = _make_agent_response()
        mock_create_tracer.return_value = MagicMock()

        inputs = {
            "task_id": "task1",
            "question": "test",
            "file_name": "",
            "file_path": "",
        }

        run_agent_for_dataset_item(inputs)

        mock_create_tracer.assert_called_once()

    @patch("evaluate_agent_on_dataset.invoke_agent_with_user_message")
    @patch("evaluate_agent_on_dataset.create_langsmith_tracer")
    def test_passes_tracing_handler_to_invoke_agent(
        self, mock_create_tracer, mock_invoke
    ):
        mock_tracer = MagicMock()
        mock_create_tracer.return_value = mock_tracer
        mock_invoke.return_value = _make_agent_response()

        inputs = {
            "task_id": "task1",
            "question": "test",
            "file_name": "",
            "file_path": "",
        }

        run_agent_for_dataset_item(inputs)

        _, kwargs = mock_invoke.call_args
        assert kwargs["tracing_handler"] is mock_tracer

    @patch("evaluate_agent_on_dataset.invoke_agent_with_user_message")
    @patch("evaluate_agent_on_dataset.create_langsmith_tracer")
    def test_passes_file_path_to_invoke_agent(
        self, mock_create_tracer, mock_invoke
    ):
        mock_invoke.return_value = _make_agent_response()
        mock_create_tracer.return_value = MagicMock()

        inputs = {
            "task_id": "task1",
            "question": "q",
            "file_name": "abc.png",
            "file_path": "2023/validation/abc.png",
        }

        run_agent_for_dataset_item(inputs)

        _, kwargs = mock_invoke.call_args
        assert kwargs["available_file_path"] == "2023/validation/abc.png"

    @patch("evaluate_agent_on_dataset.invoke_agent_with_user_message")
    @patch("evaluate_agent_on_dataset.create_langsmith_tracer")
    def test_passes_none_file_path_when_empty(
        self, mock_create_tracer, mock_invoke
    ):
        mock_invoke.return_value = _make_agent_response()
        mock_create_tracer.return_value = MagicMock()

        inputs = {
            "task_id": "task1",
            "question": "q",
            "file_name": "",
            "file_path": "",
        }

        run_agent_for_dataset_item(inputs)

        _, kwargs = mock_invoke.call_args
        assert kwargs["available_file_path"] is None


class TestEvaluateAgentOnDataset:
    @patch("evaluate_agent_on_dataset.Client")
    def test_calls_client_evaluate(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        evaluate_agent_on_dataset(
            dataset_name="my-dataset",
            name="run-1",
            description="test run",
        )

        mock_client.evaluate.assert_called_once()
        _, kwargs = mock_client.evaluate.call_args
        assert kwargs["data"] == "my-dataset"
        assert kwargs["experiment_prefix"] == "run-1"
        assert kwargs["description"] == "test run"
        assert kwargs["max_concurrency"] == 1
        assert len(kwargs["evaluators"]) == 5

    @patch("evaluate_agent_on_dataset.Client")
    def test_uses_default_experiment_prefix_when_name_empty(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        evaluate_agent_on_dataset(
            dataset_name="my-dataset",
            name="",
            description="",
        )

        _, kwargs = mock_client.evaluate.call_args
        assert kwargs["experiment_prefix"] == "gaia-eval"
