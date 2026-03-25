from pathlib import Path

import yaml

_PROMPTS_FILE = Path(__file__).parent / "prompts.yaml"

with open(_PROMPTS_FILE) as _f:
    _PROMPTS = yaml.safe_load(_f)


def get_prompt(key: str) -> str:
    """Load a prompt by key from prompts.yaml."""
    if key not in _PROMPTS:
        raise KeyError(f"Prompt '{key}' not found in {_PROMPTS_FILE.name}")
    return _PROMPTS[key].strip()
