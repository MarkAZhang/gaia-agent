from langgraph.graph import MessagesState


def return_llm_refusal(state: MessagesState) -> dict:
    return {"messages": [{"role": "ai", "content": "LLM refused to continue"}]}
