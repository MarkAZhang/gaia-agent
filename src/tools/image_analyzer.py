"""Image analysis tool powered by LLMs."""

from abc import ABC, abstractmethod

from langchain_core.tools import BaseTool, tool

from agent_graph.file_paths import to_local_file_path
from tools.tool_response import ToolError, ToolSuccess


# In order to make it easy to switch LLM models and providers, we define
# the ImageAnalyzer implementation elsewhere in the llm_wrappers module.
# This file has no dependency on an LLM-specific library, and does not need to change
# if we switch LLMs.
class BaseImageAnalyzer(ABC):
    @abstractmethod
    def answer_image_question(self, local_file_path: str, question: str) -> str:
        pass


def create_image_analyzer_tool(analyzer: BaseImageAnalyzer) -> BaseTool:
    """Return an ``analyze_image`` tool that uses the given *analyzer*.

    Parameters
    ----------
    analyzer:
        The analyzer to use.
    """

    @tool
    def analyze_image(file_path: str, question: str) -> ToolSuccess | ToolError:
        """Analyze an image and answer a question about it.

        Provide an agent-facing file path (for example the one listed under
        "Provided file path" in the system prompt). The tool resolves it to
        the real location on disk, sends the image to a Gemini vision model
        along with the supplied *question*, and returns the model's answer.

        Returns a ToolSuccess with a ``response`` field containing the answer,
        or a ToolError with an ``error`` field if something goes wrong.
        """
        try:
            local_path = to_local_file_path(file_path)
            answer = analyzer.answer_image_question(
                local_file_path=local_path, question=question
            )
            return ToolSuccess(response=answer)
        except Exception as e:
            return ToolError(error=str(e))

    return analyze_image
