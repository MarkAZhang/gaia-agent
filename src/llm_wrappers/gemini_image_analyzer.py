from google import genai
from google.genai import types
import mimetypes
from tools.image_analyzer import BaseImageAnalyzer


class GeminiImageAnalyzer(BaseImageAnalyzer):
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        # Here, we use genai.Client directly, instead of using Langgraph's ChatGoogleGenerativeAI.
        # This ensures that the calls to the Gemini API are separated from the core MessagesState.
        self.client = genai.Client(api_key=self.api_key)

    def answer_image_question(self, local_file_path: str, question: str) -> str:
        mime_type, _ = mimetypes.guess_type(local_file_path)
        if mime_type is None:
            mime_type = "application/octet-stream"

        with open(local_file_path, "rb") as f:
            image_bytes = f.read()

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                question,
            ],
        )
        return response.text
