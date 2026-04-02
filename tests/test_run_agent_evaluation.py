from unittest.mock import MagicMock, patch

from agent.agent_response import AgentResponse, AgentRunMetrics
from run_agent_evaluation import run_agent_for_dataset_item_task


def _make_agent_response(**overrides):
    defaults = {
        "answer": "answer",
        "metrics": AgentRunMetrics(
            latency_seconds=1.0,
            input_tokens=0,
            output_tokens=0,
            total_turns=0,
        ),
    }
    defaults.update(overrides)
    return AgentResponse(**defaults)


class TestRunAgentForDatasetItemTask:
    @patch("run_agent_evaluation.invoke_agent_with_user_message")
    @patch("run_agent_evaluation.CallbackHandler")
    def test_passes_question_to_invoke_agent(
        self, mock_callback_handler_cls, mock_invoke
    ):
        mock_invoke.return_value = _make_agent_response(
            metrics=AgentRunMetrics(
                latency_seconds=1.0,
                input_tokens=10,
                output_tokens=5,
                total_turns=1,
            ),
        )
        mock_callback_handler_cls.return_value = MagicMock()

        item = MagicMock()
        item.input = {
            "task_id": "task1",
            "question": "What is 2+2?",
            "file_name": "",
            "file_path": "",
        }

        run_agent_for_dataset_item_task(item=item)

        mock_invoke.assert_called_once_with(
            "What is 2+2?",
            langfuse_handler=mock_callback_handler_cls.return_value,
        )

    @patch("run_agent_evaluation.invoke_agent_with_user_message")
    @patch("run_agent_evaluation.CallbackHandler")
    def test_returns_agent_response(self, mock_callback_handler_cls, mock_invoke):
        expected = _make_agent_response(
            answer="the answer is 4",
            metrics=AgentRunMetrics(
                latency_seconds=2.5,
                input_tokens=100,
                output_tokens=50,
                total_turns=3,
            ),
        )
        mock_invoke.return_value = expected
        mock_callback_handler_cls.return_value = MagicMock()

        item = MagicMock()
        item.input = {
            "task_id": "task1",
            "question": "What is 2+2?",
            "file_name": "",
            "file_path": "",
        }

        result = run_agent_for_dataset_item_task(item=item)

        assert result is expected

    @patch("run_agent_evaluation.invoke_agent_with_user_message")
    @patch("run_agent_evaluation.CallbackHandler")
    def test_creates_langfuse_callback_handler(
        self, mock_callback_handler_cls, mock_invoke
    ):
        mock_invoke.return_value = _make_agent_response()

        item = MagicMock()
        item.input = {
            "task_id": "task1",
            "question": "test",
            "file_name": "",
            "file_path": "",
        }

        run_agent_for_dataset_item_task(item=item)

        mock_callback_handler_cls.assert_called_once()

    @patch("run_agent_evaluation.invoke_agent_with_user_message")
    @patch("run_agent_evaluation.CallbackHandler")
    def test_passes_langfuse_handler_to_invoke_agent(
        self, mock_callback_handler_cls, mock_invoke
    ):
        mock_handler = MagicMock()
        mock_callback_handler_cls.return_value = mock_handler
        mock_invoke.return_value = _make_agent_response()

        item = MagicMock()
        item.input = {
            "task_id": "task1",
            "question": "test",
            "file_name": "",
            "file_path": "",
        }

        run_agent_for_dataset_item_task(item=item)

        _, kwargs = mock_invoke.call_args
        assert kwargs["langfuse_handler"] is mock_handler
