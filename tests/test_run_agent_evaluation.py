from unittest.mock import MagicMock, patch

from run_agent_evaluation import run_agent_for_dataset_item_task


class TestRunAgentForDatasetItemTask:
    @patch("run_agent_evaluation.invoke_agent_with_user_message")
    @patch("run_agent_evaluation.CallbackHandler")
    def test_passes_question_to_invoke_agent(
        self, mock_callback_handler_cls, mock_invoke
    ):
        mock_invoke.return_value = "answer"
        mock_callback_handler_cls.return_value = MagicMock()

        item = MagicMock()
        item.input = {
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
    def test_returns_agent_result(self, mock_callback_handler_cls, mock_invoke):
        mock_invoke.return_value = "the answer is 4"
        mock_callback_handler_cls.return_value = MagicMock()

        item = MagicMock()
        item.input = {
            "question": "What is 2+2?",
            "file_name": "",
            "file_path": "",
        }

        result = run_agent_for_dataset_item_task(item=item)

        assert result == "the answer is 4"

    @patch("run_agent_evaluation.invoke_agent_with_user_message")
    @patch("run_agent_evaluation.CallbackHandler")
    def test_creates_langfuse_callback_handler(
        self, mock_callback_handler_cls, mock_invoke
    ):
        mock_invoke.return_value = "answer"

        item = MagicMock()
        item.input = {
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
        mock_invoke.return_value = "answer"

        item = MagicMock()
        item.input = {
            "question": "test",
            "file_name": "",
            "file_path": "",
        }

        run_agent_for_dataset_item_task(item=item)

        _, kwargs = mock_invoke.call_args
        assert kwargs["langfuse_handler"] is mock_handler
