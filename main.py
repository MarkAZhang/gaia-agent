import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableConfig

from agent.deps import AgentDeps
from agent.graph import build_graph
from tools.web_search import create_web_search
from langfuse import get_client


def _create_langfuse_handler():
    """Create a Langfuse callback handler if USE_LANGFUSE is enabled."""
    if os.environ.get("USE_LANGFUSE") != "1":
        return None

    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


def main():
    load_dotenv()

    langfuse_handler = _create_langfuse_handler()

    tools = [create_web_search()]
    llm = ChatAnthropic(model="claude-opus-4-6").bind_tools(tools)
    config = RunnableConfig(
        configurable={"deps": AgentDeps(llm=llm)},
        callbacks=[langfuse_handler] if langfuse_handler else [],
    )
    graph = build_graph(tools=tools)
    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Return hello world",
                }
            ]
        },
        config=config,
    )
    for msg in result["messages"]:
        print(f"[{msg.type}] {msg.content}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  -> tool_call: {tc['name']}({tc['args']})")

    if langfuse_handler:
        langfuse = get_client()
        langfuse.flush()


if __name__ == "__main__":
    main()
