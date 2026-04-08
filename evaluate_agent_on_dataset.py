from typing import TypedDict

from langfuse import get_client
from agent.agent_response import AgentResponse
from langfuse.langchain import CallbackHandler
from langfuse.api import DatasetItem
from agent.invoke_agent_with_user_message import invoke_agent_with_user_message

from evaluators.gaia_score_evaluator import gaia_score_evaluator
from evaluators.metrics.latency_evaluator import latency_evaluator
from evaluators.metrics.token_usage_evaluator import (
    input_tokens_evaluator,
    output_tokens_evaluator,
)
from evaluators.metrics.total_turns_evaluator import total_turns_evaluator


class DatasetItemInput(TypedDict):
    """
    The input for a dataset item from Langfuse.

    Attributes:
        question: The question to be answered by the agent.
        file_name: The name of the file associated with the question.
        file_path: The path to the file associated with the question.
    """

    task_id: str
    question: str
    file_name: str
    file_path: str


def run_agent_for_dataset_item_task(*, item: DatasetItem) -> AgentResponse:
    """
    Run agent for a single dataset item.

    Args:
        item: The dataset item containing input and expected output.
    """
    input: DatasetItemInput = item.input
    langfuse_handler = CallbackHandler()
    print("Running agent for task_id:", input["task_id"])

    return invoke_agent_with_user_message(
        input["question"],
        langfuse_handler=langfuse_handler,
        available_file_path=input.get("file_path") or None,
    )


def evaluate_agent_on_dataset(dataset_name: str, name: str, description: str):
    langfuse = get_client()
    dataset = langfuse.get_dataset(dataset_name)

    dataset.run_experiment(
        name=name,
        description=description,
        task=run_agent_for_dataset_item_task,
        evaluators=[
            gaia_score_evaluator,
            latency_evaluator,
            input_tokens_evaluator,
            output_tokens_evaluator,
            total_turns_evaluator,
        ],
    )

    langfuse.flush()
