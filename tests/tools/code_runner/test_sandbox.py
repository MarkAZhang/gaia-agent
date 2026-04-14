from unittest.mock import MagicMock, patch

import pytest

from tools.code_runner import sandbox as sandbox_mod
from tools.code_runner.sandbox import get_sandbox, shutdown_sandbox


@pytest.fixture(autouse=True)
def _reset_sandbox():
    """Reset the sandbox singleton before and after each test."""
    sandbox_mod._sandbox = None
    yield
    sandbox_mod._sandbox = None


def test_get_sandbox_creates_sandbox_on_first_call():
    mock_sandbox = MagicMock()
    with patch.object(sandbox_mod, "Sandbox") as mock_cls:
        mock_cls.create.return_value = mock_sandbox
        result = get_sandbox()

    assert result is mock_sandbox
    mock_cls.create.assert_called_once_with(timeout=3600)


def test_get_sandbox_returns_same_instance_on_subsequent_calls():
    mock_sandbox = MagicMock()
    with patch.object(sandbox_mod, "Sandbox") as mock_cls:
        mock_cls.create.return_value = mock_sandbox
        first = get_sandbox()
        second = get_sandbox()

    assert first is second
    mock_cls.create.assert_called_once()


def test_shutdown_sandbox_kills_existing_sandbox():
    mock_sandbox = MagicMock()
    sandbox_mod._sandbox = mock_sandbox

    shutdown_sandbox()

    mock_sandbox.kill.assert_called_once()
    assert sandbox_mod._sandbox is None


def test_shutdown_sandbox_noop_when_no_sandbox():
    shutdown_sandbox()
    assert sandbox_mod._sandbox is None


def test_get_sandbox_after_shutdown_creates_new_sandbox():
    first_sandbox = MagicMock()
    second_sandbox = MagicMock()

    with patch.object(sandbox_mod, "Sandbox") as mock_cls:
        mock_cls.create.return_value = first_sandbox
        get_sandbox()

    sandbox_mod._sandbox = None  # simulate shutdown

    with patch.object(sandbox_mod, "Sandbox") as mock_cls:
        mock_cls.create.return_value = second_sandbox
        result = get_sandbox()

    assert result is second_sandbox
