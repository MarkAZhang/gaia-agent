import os

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from openai import OpenAI

from agent_graph.prompts.get_prompt import get_prompt

OUTPUT_FORMATTER_MODEL = "gpt-4o-mini"


def _find_user_question(messages: list) -> str:
    for msg in messages:
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else ""
    return ""


def output_formatter(state: MessagesState) -> dict:
    final_answer = state["messages"][-1].content
    user_question = _find_user_question(state["messages"])
    system_prompt = get_prompt("output_formatter_system_prompt")

    user_prompt = (
        f"Original question: {user_question}\n\n"
        f"Final answer: {final_answer}"
    )

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=OUTPUT_FORMATTER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    formatted_answer = response.choices[0].message.content
    if formatted_answer is None:
        formatted_answer = final_answer

    return {"messages": [{"role": "ai", "content": formatted_answer}]}
