from unittest.mock import patch

from scripts.evaluate_agent import main


class TestEvaluateAgentScript:
    @patch("scripts.evaluate_agent.evaluate_agent_on_dataset")
    @patch("scripts.evaluate_agent.load_dotenv")
    def test_passes_evaluation_set_as_dataset_name(
        self, mock_load_dotenv, mock_evaluate
    ):
        with patch(
            "sys.argv",
            ["evaluate_agent", "my-dataset"],
        ):
            main()

        mock_evaluate.assert_called_once_with(
            dataset_name="my-dataset",
            name="",
            description="",
        )

    @patch("scripts.evaluate_agent.evaluate_agent_on_dataset")
    @patch("scripts.evaluate_agent.load_dotenv")
    def test_passes_name_and_description(self, mock_load_dotenv, mock_evaluate):
        with patch(
            "sys.argv",
            [
                "evaluate_agent",
                "my-dataset",
                "--name",
                "run-1",
                "--description",
                "test run",
            ],
        ):
            main()

        mock_evaluate.assert_called_once_with(
            dataset_name="my-dataset",
            name="run-1",
            description="test run",
        )

    @patch("scripts.evaluate_agent.evaluate_agent_on_dataset")
    @patch("scripts.evaluate_agent.load_dotenv")
    def test_loads_dotenv(self, mock_load_dotenv, mock_evaluate):
        with patch("sys.argv", ["evaluate_agent", "ds"]):
            main()

        mock_load_dotenv.assert_called_once()


class TestEvaluateAgentScriptDefaults:
    @patch("scripts.evaluate_agent.evaluate_agent_on_dataset")
    @patch("scripts.evaluate_agent.load_dotenv")
    def test_name_defaults_to_empty_string(self, mock_load_dotenv, mock_evaluate):
        with patch("sys.argv", ["evaluate_agent", "ds"]):
            main()

        _, kwargs = mock_evaluate.call_args
        assert kwargs["name"] == ""

    @patch("scripts.evaluate_agent.evaluate_agent_on_dataset")
    @patch("scripts.evaluate_agent.load_dotenv")
    def test_description_defaults_to_empty_string(
        self, mock_load_dotenv, mock_evaluate
    ):
        with patch("sys.argv", ["evaluate_agent", "ds"]):
            main()

        _, kwargs = mock_evaluate.call_args
        assert kwargs["description"] == ""
