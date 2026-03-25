from dotenv import load_dotenv

from agent.graph import build_graph


def main():
    load_dotenv()
    graph = build_graph()
    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Use the noop tool with input 'hello'. Return the response as a single word.",
                }
            ]
        }
    )
    for msg in result["messages"]:
        print(f"[{msg.type}] {msg.content}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  -> tool_call: {tc['name']}({tc['args']})")


if __name__ == "__main__":
    main()
