from pathlib import Path
from typing import Optional

from e2b_code_interpreter import Sandbox
from langchain_core.tools import tool


def _run_in_sandbox(code: str, language: Optional[str]) -> str:
    with Sandbox() as sandbox:
        execution = sandbox.run_code(code, language=language or "python")

    parts: list[str] = []
    if execution.logs.stdout:
        parts.append("stdout:\n" + "".join(execution.logs.stdout))
    if execution.logs.stderr:
        parts.append("stderr:\n" + "".join(execution.logs.stderr))
    if execution.error:
        parts.append(
            f"error: {execution.error.name}: {execution.error.value}"
        )
    for result in execution.results:
        if getattr(result, "text", None):
            parts.append(f"result: {result.text}")
    return "\n".join(parts) if parts else "(no output)"


@tool
def execute_code_snippet(snippet: str, language: Optional[str] = "python") -> str:
    """Execute a code snippet in a secure E2B sandbox and return its output.

    Use this when you need to run a short piece of code to compute a result,
    test logic, or process data. Defaults to Python; pass `language` to use
    another supported language (e.g. "javascript", "r", "bash").
    """
    return _run_in_sandbox(snippet, language)


@tool
def execute_code_file(file_name: str, language: Optional[str] = "python") -> str:
    """Execute a local code file in a secure E2B sandbox and return its output.

    Provide the path to a file on the local filesystem; its contents will be
    read and executed in the sandbox. Defaults to Python; pass `language` to
    use another supported language (e.g. "javascript", "r", "bash").
    """
    code = Path(file_name).read_text()
    return _run_in_sandbox(code, language)
