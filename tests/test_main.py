from unittest.mock import MagicMock, patch

from main import _create_langfuse_handler


class TestCreateLangfuseHandler:
    def test_returns_none_when_use_langfuse_not_set(self):
        with patch.dict("os.environ", {}, clear=True):
            assert _create_langfuse_handler() is None

    def test_returns_none_when_use_langfuse_is_zero(self):
        with patch.dict("os.environ", {"USE_LANGFUSE": "0"}, clear=True):
            assert _create_langfuse_handler() is None

    def test_returns_none_when_use_langfuse_is_empty(self):
        with patch.dict("os.environ", {"USE_LANGFUSE": ""}, clear=True):
            assert _create_langfuse_handler() is None

    @patch("langfuse.langchain.CallbackHandler")
    def test_returns_handler_when_use_langfuse_is_1(self, mock_handler_cls):
        mock_handler_cls.return_value = MagicMock()
        with patch.dict(
            "os.environ",
            {"USE_LANGFUSE": "1"},
            clear=True,
        ):
            handler = _create_langfuse_handler()
            assert handler is not None
            mock_handler_cls.assert_called_once_with(environment=None)

    @patch("langfuse.langchain.CallbackHandler")
    def test_passes_environment_to_handler(self, mock_handler_cls):
        mock_handler_cls.return_value = MagicMock()
        with patch.dict(
            "os.environ",
            {
                "USE_LANGFUSE": "1",
                "LANGFUSE_TRACING_ENVIRONMENT": "staging",
            },
            clear=True,
        ):
            handler = _create_langfuse_handler()
            assert handler is not None
            mock_handler_cls.assert_called_once_with(environment="staging")
