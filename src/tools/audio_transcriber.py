"""Audio transcription tool powered by faster-whisper."""

from faster_whisper import WhisperModel
from langchain_core.tools import tool

from agent_graph.file_paths import to_local_file_path

model = WhisperModel("turbo", device="cpu", compute_type="int8")


@tool
def transcribe_audio(file_path: str) -> str:
    """Transcribe an audio file and return the transcript as text.

    Provide an agent-facing file path (for example the one listed under
    "Provided file path" in the system prompt). The tool resolves it to the
    real location on disk, transcribes the audio using Whisper, and returns
    the full transcript.
    """
    local_path = to_local_file_path(file_path)
    segments, _ = model.transcribe(local_path)
    return " ".join(segment.text.strip() for segment in segments)
