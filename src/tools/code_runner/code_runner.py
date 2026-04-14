import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from agent_graph.file_paths import to_local_file_path
from tools.code_runner.sandbox import get_sandbox
from tools.tool_response import ToolError, ToolSuccess


@dataclass
class CodeExecutionResult:
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    results: list[str] = field(default_factory=list)

    def as_dict(self) -> str:
        return json.dumps(asdict(self))


def _run_in_sandbox(code: str, language: Optional[str], timeout: int = 60) -> str:
    sandbox = get_sandbox()
    execution = sandbox.run_code(code, language=language or "python", timeout=timeout)

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
def execute_code_snippet(
    snippet: str, language: Optional[str] = "python", timeout: int = 60
) -> ToolSuccess | ToolError:
    """Execute a code snippet in a secure, persistent E2B sandbox and return its output.

    Use this when you need to run a short piece of code to compute a result,
    test logic, or process data. Defaults to Python; pass `language` to use
    another supported language (e.g. "javascript", "r", "bash"). Returns a
    JSON string with stdout, stderr, error, and results fields.

    Note that this sandbox is persistent across multiple calls to the tool, so
    packages you install will be available in subsequent calls.

    The ``timeout`` parameter controls how long the code is allowed to run (in
    seconds, default 60). You may increase this only when necessary to solve
    the problem. If the code times out with the default timeout, first check
    whether the code is accidentally doing more work than necessary (e.g.
    grepping the entire file system or running an overly large simulation) and
    whether a more efficient approach exists (e.g. an analytical solution
    instead of a brute-force simulation) before raising the timeout.

    Returns a ToolSuccess with a ``response`` field containing the JSON result,
    or a ToolError with an ``error`` field if something goes wrong.
    """
    try:
        return ToolSuccess(response=_run_in_sandbox(snippet, language, timeout))
    except Exception as e:
        return ToolError(error=str(e))


@tool
def execute_code_file(
    file_path: str, language: Optional[str] = "python", timeout: int = 60
) -> ToolSuccess | ToolError:
    """Execute a local code file in a secure, persistent E2B sandbox and return its output.

    Provide an agent-facing file path (for example the one listed under
    "Provided file path" in the system prompt). The tool resolves it
    to the real location on disk, reads the contents, and executes them
    in the sandbox. Defaults to Python; pass `language` to use another
    supported language (e.g. "javascript", "r", "bash"). Returns a JSON
    string with stdout, stderr, error, and results fields.

    Note that this sandbox is persistent across multiple calls to the tool, so
    packages you install will be available in subsequent calls.

    The ``timeout`` parameter controls how long the code is allowed to run (in
    seconds, default 60). You may increase this only when necessary to solve
    the problem. If the code times out with the default timeout, first check
    whether the code is accidentally doing more work than necessary (e.g.
    grepping the entire file system or running an overly large simulation) and
    whether a more efficient approach exists (e.g. an analytical solution
    instead of a brute-force simulation) before raising the timeout.

    Returns a ToolSuccess with a ``response`` field containing the JSON result,
    or a ToolError with an ``error`` field if something goes wrong.
    """
    try:
        local_path = to_local_file_path(file_path)
        code = Path(local_path).read_text()
        return ToolSuccess(response=_run_in_sandbox(code, language, timeout))
    except Exception as e:
        return ToolError(error=str(e))
