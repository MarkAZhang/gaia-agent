from dotenv import load_dotenv

from agent.invoke_agent_with_user_message import build_agent_graph_and_config


def main():
    load_dotenv()
    compiled_graph_and_config = build_agent_graph_and_config(langfuse_handler=None)
    print(compiled_graph_and_config.graph.get_graph().draw_mermaid())


if __name__ == "__main__":
    main()
