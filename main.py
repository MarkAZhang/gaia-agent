from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableConfig

from agent.deps import AgentDeps
from agent.graph import TOOLS, build_graph


def main():
    load_dotenv()
    llm = ChatAnthropic(model="claude-opus-4-6").bind_tools(TOOLS)
    config = RunnableConfig(configurable={"deps": AgentDeps(llm=llm)})
    graph = build_graph()
    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Use the noop tool with input 'hello'. Return the response as a single word.",
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
