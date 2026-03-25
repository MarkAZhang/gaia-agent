import os

from dotenv import load_dotenv
from langfuse import get_client

from agent.invoke_agent_with_user_message import invoke_agent_with_user_message


def _create_langfuse_handler():
    """Create a Langfuse callback handler if USE_LANGFUSE is enabled."""
    if os.environ.get("USE_LANGFUSE") != "1":
        return None

    from langfuse.langchain import CallbackHandler

    environment = os.environ.get("LANGFUSE_TRACING_ENVIRONMENT")
    return CallbackHandler(environment=environment)


def main():
    load_dotenv()

    langfuse_handler = _create_langfuse_handler()

    result = invoke_agent_with_user_message("Return hello world", langfuse_handler)
    print(result)

    if langfuse_handler:
        langfuse = get_client()
        langfuse.flush()


if __name__ == "__main__":
    main()
