from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from agent.edges.check_answer_routing import check_answer_routing


def test_routes_to_llm_call_when_last_message_is_human():
    msg = HumanMessage(content="Error: please format correctly.")
    state = {"messages": [msg]}
    assert check_answer_routing(state) == "llm_call"


def test_routes_to_end_when_last_message_is_ai():
    msg = AIMessage(content="42")
    state = {"messages": [msg]}
    assert check_answer_routing(state) == END


def test_uses_last_message_type():
    first = HumanMessage(content="Error message")
    last = AIMessage(content="Final answer")
    state = {"messages": [first, last]}
    assert check_answer_routing(state) == END
