import os

from dotenv import load_dotenv
from langfuse import get_client

from agent.invoke_agent_with_user_message import invoke_agent_with_user_message


def _create_langfuse_handler():
    """
    Create a Langfuse callback handler if USE_LANGFUSE is enabled.

    NOTE: LANGFUSE_TRACING_ENVIRONMENT and other Langfuse-related
    environment variables are automatically picked up by Langfuse.
    """
    if os.environ.get("USE_LANGFUSE") != "1":
        return None

    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


def run_with_custom_user_message(user_message: str):
    load_dotenv()

    langfuse_handler = _create_langfuse_handler()

    result = invoke_agent_with_user_message(user_message, langfuse_handler)
    print(result.answer)

    if langfuse_handler:
        langfuse = get_client()
        langfuse.flush()


if __name__ == "__main__":
    user_message = "Return hello world"
    run_with_custom_user_message(user_message)
