import argparse

from dotenv import load_dotenv

from evaluate_agent_on_dataset import evaluate_agent_on_dataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate agent on a dataset")
    parser.add_argument(
        "evaluation_set",
        help="Name of the Langfuse evaluation dataset",
    )
    parser.add_argument(
        "--name",
        default="",
        help="Name for the evaluation run",
    )
    parser.add_argument(
        "--description",
        default="",
        help="Description for the evaluation run",
    )
    args = parser.parse_args()

    load_dotenv()
    evaluate_agent_on_dataset(
        dataset_name=args.evaluation_set,
        name=args.name,
        description=args.description,
    )


if __name__ == "__main__":
    main()
