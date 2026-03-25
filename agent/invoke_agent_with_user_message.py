from langchain_anthropic import ChatAnthropic
from tools.web_search import create_web_search

from langchain_core.runnables import RunnableConfig

from agent.deps import AgentDeps
from agent.graph import build_graph


def invoke_agent_with_user_message(input_str, langfuse_handler) -> str:
    tools = [create_web_search()]
    llm = ChatAnthropic(model="claude-opus-4-6").bind_tools(tools)
    config = RunnableConfig(
        configurable={"deps": AgentDeps(llm=llm)},
        callbacks=[langfuse_handler] if langfuse_handler else [],
    )
    graph = build_graph(tools=tools)

    result = graph.invoke(
        {"messages": [{"role": "user", "content": input_str}]},
        config=config,
    )

    final_answer = (
        result["messages"][-1].content if result["messages"] else "No answer found"
    )

    return final_answer
