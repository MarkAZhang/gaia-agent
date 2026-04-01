from langgraph.graph import MessagesState


def return_llm_tool_not_available(state: MessagesState) -> dict:
    last_message = state["messages"][-1]
    content = last_message.content if isinstance(last_message.content, str) else ""
    last_line = content.strip().splitlines()[-1] if content.strip() else ""
    return {"messages": [{"role": "ai", "content": last_line}]}
