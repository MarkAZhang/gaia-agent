import argparse

from dotenv import load_dotenv

from evaluate_agent_on_dataset import evaluate_agent_on_dataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate agent on a dataset")
    parser.add_argument(
        "evaluation_set",
        help="Name of the LangSmith evaluation dataset",
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
    parser.add_argument(
        "--example-ids",
        nargs="+",
        default=None,
        help=(
            "Optional LangSmith example IDs to evaluate within the dataset "
            "(defaults to all examples)"
        ),
    )
    args = parser.parse_args()

    load_dotenv()
    evaluate_agent_on_dataset(
        dataset_name=args.evaluation_set,
        name=args.name,
        description=args.description,
        example_ids=args.example_ids,
    )


if __name__ == "__main__":
    main()
