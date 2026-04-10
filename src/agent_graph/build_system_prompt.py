from typing import Optional

from agent_graph.prompts.get_prompt import get_prompt


def _build_file_path_block(available_file_path: Optional[str]) -> str:
    """Render the file-path section injected into the system prompt.

    Returns an empty string when no path was provided so the conclusion
    section renders cleanly.
    """
    if not available_file_path:
        return ""
    return f"Provided file path:\n- {available_file_path}\n\n"


def build_system_prompt(
    available_file_path: Optional[str] = None,
) -> str:
    return get_prompt(
        "react_system_prompt",
        file_paths_block=_build_file_path_block(available_file_path),
    )
