from unittest.mock import MagicMock, patch

from scripts.print_mermaid_for_agent_graph import main


class TestPrintMermaidScript:
    @patch("scripts.print_mermaid_for_agent_graph.build_agent_graph_and_config")
    @patch("scripts.print_mermaid_for_agent_graph.load_dotenv")
    def test_calls_build_agent_graph_with_no_langfuse(
        self, mock_load_dotenv, mock_build
    ):
        mock_graph = MagicMock()
        mock_build.return_value = mock_graph

        main()

        mock_build.assert_called_once_with(langfuse_handler=None)

    @patch("scripts.print_mermaid_for_agent_graph.build_agent_graph_and_config")
    @patch("scripts.print_mermaid_for_agent_graph.load_dotenv")
    def test_prints_mermaid_output(self, mock_load_dotenv, mock_build, capsys):
        mock_compiled = MagicMock()
        mock_compiled.graph.get_graph.return_value.draw_mermaid.return_value = (
            "graph TD; A-->B"
        )
        mock_build.return_value = mock_compiled

        main()

        captured = capsys.readouterr()
        assert "graph TD; A-->B" in captured.out

    @patch("scripts.print_mermaid_for_agent_graph.build_agent_graph_and_config")
    @patch("scripts.print_mermaid_for_agent_graph.load_dotenv")
    def test_loads_dotenv(self, mock_load_dotenv, mock_build):
        mock_build.return_value = MagicMock()

        main()

        mock_load_dotenv.assert_called_once()
