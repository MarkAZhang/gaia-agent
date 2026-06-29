from dotenv import load_dotenv

from agent_graph.invoke_agent_with_user_message import invoke_agent_with_user_message
from observability.tracing import create_tracing_handler_if_enabled


def run_with_custom_user_message(user_message: str):
    load_dotenv()

    tracing_handler = create_tracing_handler_if_enabled()

    result = invoke_agent_with_user_message(user_message, tracing_handler)
    print(result.answer)


if __name__ == "__main__":
    user_message = "Return hello world"
    run_with_custom_user_message(user_message)
