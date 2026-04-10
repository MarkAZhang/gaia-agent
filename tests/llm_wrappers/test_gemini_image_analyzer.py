from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch


def _make_analyzer(model="gemini-3.1-pro", api_key="fake-key"):
    with patch("llm_wrappers.gemini_image_analyzer.genai") as mock_genai:
        mock_genai.Client.return_value = MagicMock()
        from llm_wrappers.gemini_image_analyzer import GeminiImageAnalyzer

        analyzer = GeminiImageAnalyzer(model=model, api_key=api_key)
    return analyzer, mock_genai


def test_sends_image_bytes_and_question():
    analyzer, _ = _make_analyzer()
    analyzer.client.models.generate_content.return_value = SimpleNamespace(
        text="A cat sitting on a mat."
    )

    fake_image_data = b"\x89PNG\r\n\x1a\nfake-image-data"

    with patch("builtins.open", mock_open(read_data=fake_image_data)):
        result = analyzer.answer_image_question(
            local_file_path="cat.png", question="What is in this image?"
        )

    assert result == "A cat sitting on a mat."
    analyzer.client.models.generate_content.assert_called_once()
    call_kwargs = analyzer.client.models.generate_content.call_args
    assert call_kwargs.kwargs["model"] == "gemini-3.1-pro"
    contents = call_kwargs.kwargs["contents"]
    assert contents[1] == "What is in this image?"


def test_detects_jpeg_mime_type():
    analyzer, _ = _make_analyzer()
    analyzer.client.models.generate_content.return_value = SimpleNamespace(text="ok")

    with patch("builtins.open", mock_open(read_data=b"jpeg-data")):
        analyzer.answer_image_question(
            local_file_path="photo.jpeg", question="Describe."
        )

    call_kwargs = analyzer.client.models.generate_content.call_args
    contents = call_kwargs.kwargs["contents"]
    image_part = contents[0]
    assert image_part.inline_data.mime_type == "image/jpeg"


def test_falls_back_to_octet_stream_for_unknown_type():
    analyzer, _ = _make_analyzer()
    analyzer.client.models.generate_content.return_value = SimpleNamespace(text="ok")

    with patch("builtins.open", mock_open(read_data=b"unknown-data")):
        analyzer.answer_image_question(
            local_file_path="file.xyz123", question="What?"
        )

    call_kwargs = analyzer.client.models.generate_content.call_args
    contents = call_kwargs.kwargs["contents"]
    image_part = contents[0]
    assert image_part.inline_data.mime_type == "application/octet-stream"


def test_passes_api_key_to_client():
    _, mock_genai = _make_analyzer(api_key="my-secret-key")
    mock_genai.Client.assert_called_once_with(api_key="my-secret-key")
