from typing import TypedDict
from langfuse import get_client
from dotenv import load_dotenv
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


class DatasetItem(TypedDict):
    input: DatasetItemInput


def run_agent_for_dataset_item_task(dataset_item: DatasetItem):
    # TODO: Maybe add the Langfuse Callback Handler.
    """
    Run agent for a single dataset item.

    Args:
        dataset_item: The dataset item containing input and expected output.
    """
    input = dataset_item["input"]

    return invoke_agent_with_user_message(input["question"], langfuse_handler=None)


def evaluate_agent_on_gaia_20(version_tag):
    langfuse = get_client()
    dataset = langfuse.get_dataset("GAIA 20")

    dataset.run_experiment(
        name=version_tag,
        task=run_agent_for_dataset_item_task,
        evaluators=[gaia_score_evaluator],
    )

    langfuse.flush()


if __name__ == "__main__":
    load_dotenv()
    evaluate_agent_on_gaia_20(version_tag="2026-03-24-dry-run")
