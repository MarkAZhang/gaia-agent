from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableConfig

from agent.deps import AgentDeps
from agent.graph import build_graph
from tools.web_search import create_web_search


def main():
    load_dotenv()
    tools = [create_web_search()]
    llm = ChatAnthropic(model="claude-opus-4-6").bind_tools(tools)
    config = RunnableConfig(configurable={"deps": AgentDeps(llm=llm)})
    graph = build_graph(tools=tools)
    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Search the web for 'LangGraph framework' and summarize what you find.",
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


if __name__ == "__main__":
    main()
