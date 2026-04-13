from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, ToolMessage


class AgentGraphState(TypedDict):
    # The system message, user message, and core agent messages.
    agent_messages: Annotated[list[AnyMessage], add_messages]
    # Tool messages.
    tool_messages: Annotated[list[ToolMessage], add_messages]


def get_messages_for_core_agent(state: AgentGraphState) -> list[AnyMessage]:
    # Only pass the most recent tool message to the core agent, along with all agent messages.
    # Tool messages tend to be long.
    # The agent is asked to summarize the tool results, so it theoretically doesn't
    # need the full tool message history.
    #
    # NOTE: We are still experimenting with the best approach to balance agent accuracy and input tokens sent.
    if state["tool_messages"]:
        return state["agent_messages"] + [state["tool_messages"][-1]]
    else:
        return state["agent_messages"]
