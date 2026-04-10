import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _mock_execution(stdout=None, stderr=None, error=None, results=None):
    return SimpleNamespace(
        logs=SimpleNamespace(stdout=stdout or [], stderr=stderr or []),
        error=error,
        results=results or [],
    )


def _patch_sandbox(execution):
    sandbox_instance = MagicMock()
    sandbox_instance.run_code.return_value = execution
    cm = MagicMock()
    cm.__enter__.return_value = sandbox_instance
    cm.__exit__.return_value = False
    return patch("tools.code_runner.Sandbox.create", return_value=cm), sandbox_instance


def test_execute_code_snippet_default_language_python():
    from tools.code_runner import execute_code_snippet

    execution = _mock_execution(stdout=["hello\n"])
    patcher, sandbox = _patch_sandbox(execution)
    with patcher:
        result = execute_code_snippet.invoke({"snippet": "print('hello')"})

    sandbox.run_code.assert_called_once_with("print('hello')", language="python")
    payload = json.loads(result)
    assert payload["stdout"] == "hello\n"


def test_execute_code_snippet_custom_language():
    from tools.code_runner import execute_code_snippet

    execution = _mock_execution(
        results=[SimpleNamespace(text="42")],
    )
    patcher, sandbox = _patch_sandbox(execution)
    with patcher:
        result = execute_code_snippet.invoke(
            {"snippet": "console.log(42)", "language": "javascript"}
        )

    sandbox.run_code.assert_called_once_with(
        "console.log(42)", language="javascript"
    )
    payload = json.loads(result)
    assert payload["results"] == ["42"]


def test_execute_code_snippet_reports_error_and_stderr():
    from tools.code_runner import execute_code_snippet

    execution = _mock_execution(
        stderr=["bad\n"],
        error=SimpleNamespace(name="ValueError", value="boom"),
    )
    patcher, _ = _patch_sandbox(execution)
    with patcher:
        result = execute_code_snippet.invoke({"snippet": "raise ValueError('boom')"})

    payload = json.loads(result)
    assert payload["stderr"] == "bad\n"
    assert payload["error"] == "ValueError: boom"


def test_execute_code_snippet_no_output():
    from tools.code_runner import execute_code_snippet

    patcher, _ = _patch_sandbox(_mock_execution())
    with patcher:
        result = execute_code_snippet.invoke({"snippet": "x = 1"})

    payload = json.loads(result)
    assert payload == {"stdout": "", "stderr": "", "error": None, "results": []}


def test_execute_code_file_reads_and_runs_file():
    from tools.code_runner import execute_code_file

    execution = _mock_execution(stdout=["ok"])
    patcher, sandbox = _patch_sandbox(execution)
    with patcher, patch("tools.code_runner.Path") as mock_path:
        mock_path.return_value.read_text.return_value = "print('ok')"
        result = execute_code_file.invoke(
            {"file_path": "2023/validation/script.py"}
        )

    # The agent-facing path should be resolved to the GAIA files root
    # before the file is read from disk.
    mock_path.assert_called_once_with(
        ".gaia-questions/files/2023/validation/script.py"
    )
    sandbox.run_code.assert_called_once_with("print('ok')", language="python")
    payload = json.loads(result)
    assert payload["stdout"] == "ok"


def test_execute_code_file_custom_language():
    from tools.code_runner import execute_code_file

    patcher, sandbox = _patch_sandbox(_mock_execution(stdout=["hi"]))
    with patcher, patch("tools.code_runner.Path") as mock_path:
        mock_path.return_value.read_text.return_value = "echo hi"
        execute_code_file.invoke(
            {"file_path": "run.sh", "language": "bash"}
        )

    mock_path.assert_called_once_with(".gaia-questions/files/run.sh")
    sandbox.run_code.assert_called_once_with("echo hi", language="bash")
