from agent_graph.context.agent_graph_state import AgentGraphState
from agent_graph.prompts.get_prompt import get_prompt


def check_and_get_final_answer(state: AgentGraphState) -> dict:
    last_message = state["agent_messages"][-1]
    content = last_message.content if isinstance(last_message.content, str) else ""
    last_line = content.strip().split("\n")[-1] if content.strip() else ""

    if last_line.startswith("Ans:"):
        final_answer = last_line[len("Ans:") :].strip()
        return {"agent_messages": [{"role": "ai", "content": final_answer}]}

    error_msg = get_prompt("final_answer_format_error")
    return {"agent_messages": [{"role": "system", "content": error_msg}]}
