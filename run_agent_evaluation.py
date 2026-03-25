from typing import TypedDict
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv
from langfuse.api import DatasetItem
from agent.invoke_agent_with_user_message import invoke_agent_with_user_message

from gaia_score_evaluator import gaia_score_evaluator


class DatasetItemInput(TypedDict):
    """
    The input for a dataset item from Langfuse.

    Attributes:
        question: The question to be answered by the agent.
        file_name: The name of the file associated with the question.
        file_path: The path to the file associated with the question.
    """

    question: str
    file_name: str
    file_path: str


def run_agent_for_dataset_item_task(*, item: DatasetItem):
    """
    Run agent for a single dataset item.

    Args:
        item: The dataset item containing input and expected output.
    """
    input: DatasetItemInput = item.input
    langfuse_handler = CallbackHandler()

    return invoke_agent_with_user_message(input["question"], langfuse_handler=langfuse_handler)


def evaluate_agent_on_dataset(dataset, version_tag):
    langfuse = get_client()
    dataset = langfuse.get_dataset(dataset)

    dataset.run_experiment(
        name=version_tag,
        task=run_agent_for_dataset_item_task,
        evaluators=[gaia_score_evaluator],
    )

    langfuse.flush()


if __name__ == "__main__":
    load_dotenv()
    evaluate_agent_on_dataset(dataset="GAIA 2", version_tag="2026-03-24-dry-run")
