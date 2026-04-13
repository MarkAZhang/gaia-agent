from langchain_core.messages import AIMessage

from agent_graph.nodes.return_llm_refusal import return_llm_refusal


def test_return_llm_refusal_returns_refusal_message():
    state = {"agent_messages": [AIMessage(content="I cannot help with that.")]}
    result = return_llm_refusal(state)
    assert result == {"agent_messages": [{"role": "ai", "content": "LLM refused to continue"}]}


def test_return_llm_refusal_ignores_state_content():
    state = {"agent_messages": [AIMessage(content="Some other content")]}
    result = return_llm_refusal(state)
    assert result["agent_messages"][0]["content"] == "LLM refused to continue"
