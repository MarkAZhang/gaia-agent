from langgraph.graph import MessagesState, END


def check_answer_routing(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if last_message.type == "human":
        return "llm_call"
    return END
