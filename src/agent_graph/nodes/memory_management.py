from langchain_core.messages import ToolMessage
from langgraph.graph import MessagesState


def memory_management(state: MessagesState) -> dict:
    # The goal is to reduce input tokens. Tool messages tend to be long while
    # not providing much useful information if the result was wrong.
    # This is especially true of web_searcher, but can be true of other tools
    # like document_analyzer as well.
    # The agent is prompted to summarize the tool results, so it theoretically
    # doesn't need the raw tool message history for subsequent turns.
    # NOTE: We are still experimenting with the best approach to balance agent
    # accuracy and input tokens sent.

    messages = state["messages"]

    # Find the last AI message index
    last_ai_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].type == "ai":
            last_ai_idx = i
            break

    if last_ai_idx is None:
        return {"messages": []}

    # Replace all tool messages before the last AI message with "removed"
    replacements = []
    for i in range(last_ai_idx):
        msg = messages[i]
        if msg.type == "tool" and msg.content != "removed":
            replacements.append(
                ToolMessage(
                    content="removed",
                    tool_call_id=msg.tool_call_id,
                    id=msg.id,
                )
            )

    return {"messages": replacements}
