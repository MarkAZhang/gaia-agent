from pathlib import Path

import yaml

_PROMPTS_FILE = Path(__file__).parent / "prompts.yaml"


def get_prompt(key: str) -> str:
    """Load a prompt by key from prompts.yaml."""
    with open(_PROMPTS_FILE) as f:
        prompts = yaml.safe_load(f)
    if key not in prompts:
        raise KeyError(f"Prompt '{key}' not found in {_PROMPTS_FILE.name}")
    return prompts[key].strip()
