from unittest.mock import patch

from tools.shutdown_tools import shutdown_tools


def test_shutdown_tools_calls_shutdown_sandbox():
    with patch("tools.shutdown_tools.shutdown_sandbox") as mock_shutdown:
        shutdown_tools()

    mock_shutdown.assert_called_once()
