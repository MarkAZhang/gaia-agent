from dataclasses import dataclass


@dataclass
class AgentRunMetrics:
    """Metrics collected during an agent run."""

    total_turns: int


@dataclass
class AgentResponse:
    """Response from running the agent, including the answer and run metrics."""

    answer: str
    metrics: AgentRunMetrics
    deobfuscation_method: str
