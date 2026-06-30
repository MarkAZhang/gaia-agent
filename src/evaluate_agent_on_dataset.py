from typing import TypedDict

from langsmith import Client

from agent_graph.invoke_agent_with_user_message import invoke_agent_with_user_message
from evaluators.gaia_score_evaluator import gaia_score_evaluator
from evaluators.metrics.total_turns_evaluator import total_turns_evaluator


class DatasetItemInput(TypedDict):
    """
    The input for a dataset item from LangSmith.

    Attributes:
        question: The question to be answered by the agent.
        file_name: The name of the file associated with the question.
        file_path: The path to the file associated with the question.
    """

    task_id: str
    question: str
    file_name: str
    file_path: str


def run_agent_for_dataset_item(inputs: DatasetItemInput) -> dict:
    """
    Run agent for a single dataset item.

    Args:
        inputs: The dataset example inputs.
    """
    print("Running agent for task_id:", inputs["task_id"])

    response = invoke_agent_with_user_message(
        inputs["question"],
        available_file_path=inputs.get("file_path") or None,
    )
    return {
        "answer": response.answer,
        "metrics": {
            "total_turns": response.metrics.total_turns,
        },
        "deobfuscation_method": response.deobfuscation_method,
    }


def evaluate_agent_on_dataset(dataset_name: str, name: str, description: str):
    client = Client()
    client.evaluate(
        run_agent_for_dataset_item,
        data=dataset_name,
        evaluators=[
            gaia_score_evaluator,
            total_turns_evaluator,
        ],
        experiment_prefix=name or "gaia-eval",
        description=description,
        max_concurrency=1,
    )
