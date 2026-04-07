import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from e2b_code_interpreter import Sandbox
from langchain_core.tools import tool


@dataclass
class CodeExecutionResult:
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    results: list[str] = field(default_factory=list)

    def as_dict(self) -> str:
        return json.dumps(asdict(self))


def _run_in_sandbox(code: str, language: Optional[str]) -> str:
    with Sandbox.create() as sandbox:
        execution = sandbox.run_code(code, language=language or "python")

    result = CodeExecutionResult()
    if execution.logs.stdout:
        result.stdout = "".join(execution.logs.stdout)
    if execution.logs.stderr:
        result.stderr = "".join(execution.logs.stderr)
    if execution.error:
        result.error = f"{execution.error.name}: {execution.error.value}"
    for item in execution.results:
        if getattr(item, "text", None):
            result.results.append(item.text)
    return result.as_dict()


@tool
def execute_code_snippet(snippet: str, language: Optional[str] = "python") -> str:
    """Execute a code snippet in a secure E2B sandbox and return its output.

    Use this when you need to run a short piece of code to compute a result,
    test logic, or process data. Defaults to Python; pass `language` to use
    another supported language (e.g. "javascript", "r", "bash"). Returns a
    JSON string with stdout, stderr, error, and results fields.
    """
    return _run_in_sandbox(snippet, language)


@tool
def execute_code_file(file_name: str, language: Optional[str] = "python") -> str:
    """Execute a local code file in a secure E2B sandbox and return its output.

    Provide the path to a file on the local filesystem; its contents will be
    read and executed in the sandbox. Defaults to Python; pass `language` to
    use another supported language (e.g. "javascript", "r", "bash"). Returns a
    JSON string with stdout, stderr, error, and results fields.
    """
    code = Path(file_name).read_text()
    return _run_in_sandbox(code, language)
