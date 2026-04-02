from dataclasses import dataclass


@dataclass
class AgentResult:
    """Result from running the agent, including the answer and metrics."""

    answer: str
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    total_turns: int
