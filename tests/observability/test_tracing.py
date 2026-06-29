from unittest.mock import MagicMock, patch

from observability.tracing import (
    create_langsmith_tracer,
    create_tracing_handler_if_enabled,
)


class TestCreateTracingHandlerIfEnabled:
    @patch.dict("os.environ", {"USE_LANGSMITH": "0"}, clear=False)
    def test_returns_none_when_disabled(self):
        assert create_tracing_handler_if_enabled() is None

    @patch("observability.tracing.create_langsmith_tracer")
    @patch.dict(
        "os.environ",
        {"USE_LANGSMITH": "1", "LANGCHAIN_PROJECT": "gaia-agent"},
        clear=False,
    )
    def test_returns_tracer_when_enabled(self, mock_create_tracer):
        mock_tracer = MagicMock()
        mock_create_tracer.return_value = mock_tracer

        assert create_tracing_handler_if_enabled() is mock_tracer
        mock_create_tracer.assert_called_once()


class TestCreateLangsmithTracer:
    @patch("observability.tracing.LangChainTracer")
    @patch("observability.tracing.Client")
    @patch.dict("os.environ", {"LANGCHAIN_PROJECT": "gaia-agent"}, clear=False)
    def test_uses_langchain_project_when_no_project_name(
        self, mock_client_cls, mock_tracer_cls
    ):
        create_langsmith_tracer()

        mock_tracer_cls.assert_called_once_with(
            project_name="gaia-agent",
            client=mock_client_cls.return_value,
        )

    @patch("observability.tracing.LangChainTracer")
    @patch("observability.tracing.Client")
    def test_uses_explicit_project_name(self, mock_client_cls, mock_tracer_cls):
        create_langsmith_tracer(project_name="custom-project")

        mock_tracer_cls.assert_called_once_with(
            project_name="custom-project",
            client=mock_client_cls.return_value,
        )
