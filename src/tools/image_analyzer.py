"""Image analysis tool powered by Gemini.

Uses the Google GenAI SDK to send a local image file along with a
question to a Gemini model and returns the model's response.
"""

import mimetypes

from google import genai
from google.genai import types
from langchain_core.tools import BaseTool, tool

from agent_graph.file_paths import to_local_file_path


def create_image_analyzer(model: str, api_key: str) -> BaseTool:
    """Return an ``analyze_image`` tool that uses the given Gemini *model*.

    Parameters
    ----------
    model:
        The Gemini model name (e.g. ``"gemini-3.1-pro"``).
    api_key:
        The Gemini API key used to authenticate requests.
    """
    client = genai.Client(api_key=api_key)

    @tool
    def analyze_image(file_path: str, question: str) -> str:
        """Analyze an image and answer a question about it.

        Provide an agent-facing file path (for example the one listed under
        "Provided file path" in the system prompt). The tool resolves it to
        the real location on disk, sends the image to a Gemini vision model
        along with the supplied *question*, and returns the model's answer.
        """
        local_path = to_local_file_path(file_path)

        mime_type, _ = mimetypes.guess_type(local_path)
        if mime_type is None:
            mime_type = "application/octet-stream"

        with open(local_path, "rb") as f:
            image_bytes = f.read()

        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                question,
            ],
        )
        return response.text

    return analyze_image
