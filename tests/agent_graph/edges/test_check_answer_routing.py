from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END

from agent_graph.edges.check_answer_routing import check_answer_routing


def test_routes_to_llm_call_when_last_message_is_system():
    msg = SystemMessage(content="Error: please format correctly.")
    state = {"agent_messages": [msg]}
    assert check_answer_routing(state) == "core_agent"


def test_routes_to_end_when_last_message_is_ai():
    msg = AIMessage(content="42")
    state = {"agent_messages": [msg]}
    assert check_answer_routing(state) == END


def test_uses_last_message_type():
    first = SystemMessage(content="Error message")
    last = AIMessage(content="Final answer")
    state = {"agent_messages": [first, last]}
    assert check_answer_routing(state) == END
