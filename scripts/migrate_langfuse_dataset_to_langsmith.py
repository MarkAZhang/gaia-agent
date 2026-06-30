"""
One-time script to migrate a Langfuse dataset to LangSmith.

Requires langfuse to be installed temporarily:
    uv pip install langfuse

Usage:
    python scripts/migrate_langfuse_dataset_to_langsmith.py my-dataset --dry-run
    python scripts/migrate_langfuse_dataset_to_langsmith.py my-dataset --langsmith-dataset gaia-validation
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from langsmith import Client


def _load_langfuse_dataset(dataset_name: str):
    try:
        from langfuse import get_client
    except ImportError as exc:
        raise SystemExit(
            "langfuse is not installed. Install it temporarily with: uv pip install langfuse"
        ) from exc

    langfuse = get_client()
    return langfuse.get_dataset(dataset_name)


def _map_examples(dataset) -> list[dict]:
    examples = []
    for item in dataset.items:
        inputs = dict(item.input)
        expected_output = item.expected_output
        if isinstance(expected_output, dict):
            outputs = expected_output
        else:
            outputs = {"answer": expected_output}
        examples.append({"inputs": inputs, "outputs": outputs})
    return examples


def migrate_dataset(
    langfuse_dataset_name: str,
    langsmith_dataset_name: str,
    *,
    dry_run: bool = False,
) -> None:
    dataset = _load_langfuse_dataset(langfuse_dataset_name)
    examples = _map_examples(dataset)

    print(f"Langfuse dataset: {langfuse_dataset_name}")
    print(f"LangSmith dataset: {langsmith_dataset_name}")
    print(f"Example count: {len(examples)}")
    if examples:
        print("Sample example:")
        print(f"  inputs: {examples[0]['inputs']}")
        print(f"  outputs: {examples[0]['outputs']}")

    if dry_run:
        print("Dry run — no LangSmith dataset created.")
        return

    client = Client()
    langsmith_dataset = client.create_dataset(dataset_name=langsmith_dataset_name)
    client.create_examples(
        dataset_id=langsmith_dataset.id,
        examples=examples,
    )
    print(f"Created LangSmith dataset '{langsmith_dataset_name}' with {len(examples)} examples.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Migrate a Langfuse dataset to LangSmith"
    )
    parser.add_argument(
        "langfuse_dataset",
        help="Name of the Langfuse dataset to export",
    )
    parser.add_argument(
        "--langsmith-dataset",
        default=None,
        help="Name for the LangSmith dataset (defaults to the Langfuse dataset name)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print migration preview without creating a LangSmith dataset",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    langsmith_dataset_name = args.langsmith_dataset or args.langfuse_dataset
    migrate_dataset(
        args.langfuse_dataset,
        langsmith_dataset_name,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
