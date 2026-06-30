import os
from typing import Optional

from langchain_core.tracers.langchain import LangChainTracer
from langsmith import Client


def create_langsmith_tracer(*, project_name: str | None = None) -> LangChainTracer:
    """Create a LangChain tracer backed by the LangSmith client."""
    client = Client()
    resolved_project = project_name or os.environ.get("LANGCHAIN_PROJECT")
    return LangChainTracer(project_name=resolved_project, client=client)


def create_tracing_handler_if_enabled() -> Optional[LangChainTracer]:
    """Return a LangSmith tracer when USE_LANGSMITH=1, otherwise None."""
    if os.environ.get("USE_LANGSMITH") != "1":
        return None
    return create_langsmith_tracer()
