"""Helpers for translating between agent-facing and local file paths.

The agent is given paths relative to the GAIA questions file root (for
example ``2023/validation/abc.png``). The actual files live under
``.gaia-questions/files/`` on the local filesystem. Tools that need to
read a file should resolve the agent-facing path through
:func:`to_local_file_path` before touching disk.
"""

from pathlib import Path

GAIA_FILES_ROOT = Path(".gaia-questions/files")


def to_local_file_path(agent_file_path: str) -> str:
    """Convert an agent-facing file path to its local filesystem path.

    If the provided path already points under :data:`GAIA_FILES_ROOT`
    (or is otherwise absolute), it is returned unchanged so that callers
    can pass either form safely.
    """
    path = Path(agent_file_path)
    if path.is_absolute():
        return str(path)
    if path.parts[:2] == GAIA_FILES_ROOT.parts:
        return str(path)
    return str(GAIA_FILES_ROOT / path)
