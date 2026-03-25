from langgraph.graph import MessagesState


def should_continue(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "check_and_get_final_answer"
