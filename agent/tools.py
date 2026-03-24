from langchain_core.tools import tool


@tool
def noop_tool(input: str) -> str:
    """A placeholder tool that does nothing. Returns the input unchanged."""
    return f"noop_tool received: {input}"
