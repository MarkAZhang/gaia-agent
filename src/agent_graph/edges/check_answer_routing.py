from langgraph.graph import MessagesState


def check_answer_routing(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if last_message.type == "system":
        return "core_agent"
    return "output_formatter"
