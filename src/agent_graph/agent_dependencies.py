from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel


@dataclass
class AgentDependencies:
    core_agent_model: BaseChatModel
    image_analyzer_model: str = ""
