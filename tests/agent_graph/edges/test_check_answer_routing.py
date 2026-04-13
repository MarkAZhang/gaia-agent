from langchain_core.messages import AIMessage, SystemMessage

from agent_graph.edges.check_answer_routing import check_answer_routing


def test_routes_to_core_agent_when_last_message_is_system():
    msg = SystemMessage(content="Error: please format correctly.")
    state = {"messages": [msg]}
    assert check_answer_routing(state) == "core_agent"


def test_routes_to_output_formatter_when_last_message_is_ai():
    msg = AIMessage(content="42")
    state = {"messages": [msg]}
    assert check_answer_routing(state) == "output_formatter"


def test_uses_last_message_type():
    first = SystemMessage(content="Error message")
    last = AIMessage(content="Final answer")
    state = {"messages": [first, last]}
    assert check_answer_routing(state) == "output_formatter"
