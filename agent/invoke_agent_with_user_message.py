import time

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from agent.agent_result import AgentResult
from agent.deps import AgentDeps
from agent.graph import build_graph
from tools.web_search import create_web_search


def _compute_metrics(messages):
    """Extract token usage and turn counts from graph result messages."""
    input_tokens = 0
    output_tokens = 0
    total_turns = 0

    for msg in messages:
        if isinstance(msg, AIMessage):
            total_turns += 1
            usage = getattr(msg, "response_metadata", {}).get("usage", {})
            input_tokens += usage.get("input_tokens", 0)
            output_tokens += usage.get("output_tokens", 0)
        elif isinstance(msg, ToolMessage):
            total_turns += 1

    return input_tokens, output_tokens, total_turns


def invoke_agent_with_user_message(input_str, langfuse_handler) -> AgentResult:
    tools = [create_web_search()]
    llm = ChatAnthropic(model="claude-opus-4-6").bind_tools(tools)
    config = RunnableConfig(
        configurable={"deps": AgentDeps(llm=llm)},
        callbacks=[langfuse_handler] if langfuse_handler else [],
    )
    graph = build_graph(tools=tools)

    start_time = time.monotonic()
    result = graph.invoke(
        {"messages": [{"role": "user", "content": input_str}]},
        config=config,
    )
    latency_seconds = time.monotonic() - start_time

    messages = result["messages"]
    final_answer = messages[-1].content if messages else "No answer found"
    input_tokens, output_tokens, total_turns = _compute_metrics(messages)

    return AgentResult(
        answer=final_answer,
        latency_seconds=latency_seconds,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_turns=total_turns,
    )
