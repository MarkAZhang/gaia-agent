from pathlib import Path
from typing import Any

import yaml

_PROMPTS_FILE = Path(__file__).parent / "prompts.yaml"

with open(_PROMPTS_FILE) as _f:
    _PROMPTS = yaml.safe_load(_f)


def get_prompt(key: str, **kwargs: Any) -> str:
    """Load a prompt by key from prompts.yaml.

    If the value under ``key`` is a mapping, its child string values are
    concatenated (in insertion order) with blank lines between them to
    form the final prompt. If ``kwargs`` are provided, the resulting
    prompt is hydrated via :meth:`str.format`.
    """
    if key not in _PROMPTS:
        raise KeyError(f"Prompt '{key}' not found in {_PROMPTS_FILE.name}")

    value = _PROMPTS[key]
    if isinstance(value, dict):
        prompt = "\n\n".join(str(part).strip() for part in value.values())
    else:
        prompt = str(value).strip()

    if kwargs:
        prompt = prompt.format(**kwargs)

    return prompt
