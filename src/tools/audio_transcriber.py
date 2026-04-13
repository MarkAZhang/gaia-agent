"""Audio transcription tool powered by faster-whisper."""

from faster_whisper import WhisperModel
from langchain_core.tools import tool

from agent_graph.file_paths import to_local_file_path
from tools.tool_response import ToolError, ToolSuccess

model = WhisperModel("turbo", device="cpu", compute_type="int8")


@tool
def transcribe_audio(file_path: str) -> ToolSuccess | ToolError:
    """Transcribe an audio file and return the transcript as text.

    Provide an agent-facing file path (for example the one listed under
    "Provided file path" in the system prompt). The tool resolves it to the
    real location on disk, transcribes the audio using Whisper, and returns
    the full transcript.

    Returns a ToolSuccess with a ``response`` field containing the transcript,
    or a ToolError with an ``error`` field if something goes wrong.
    """
    try:
        local_path = to_local_file_path(file_path)
        segments, _ = model.transcribe(local_path)
        transcript = " ".join(segment.text.strip() for segment in segments)
        return ToolSuccess(response=transcript)
    except Exception as e:
        return ToolError(error=str(e))
