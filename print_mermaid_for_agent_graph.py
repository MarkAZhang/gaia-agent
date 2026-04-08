from dotenv import load_dotenv
from agent.invoke_agent_with_user_message import build_agent_graph_and_config


def print_mermaid_for_agent_graph():
    compiled_graph_and_config = build_agent_graph_and_config(langfuse_handler=None)
    print(compiled_graph_and_config.graph.get_graph().draw_mermaid())


if __name__ == "__main__":
    load_dotenv()
    print_mermaid_for_agent_graph()
