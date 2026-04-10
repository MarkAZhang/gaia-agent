from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch


def test_analyze_image_sends_image_bytes_and_question():
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = SimpleNamespace(
        text="A cat sitting on a mat."
    )

    fake_image_data = b"\x89PNG\r\n\x1a\nfake-image-data"

    with patch("tools.image_analyzer.genai") as mock_genai:
        mock_genai.Client.return_value = mock_client
        from tools.image_analyzer import create_image_analyzer

        tool = create_image_analyzer(model="gemini-3.1-pro", api_key="fake-key")

    with patch("builtins.open", mock_open(read_data=fake_image_data)):
        result = tool.invoke(
            {"file_path": "2023/validation/cat.png", "question": "What is in this image?"}
        )

    assert result == "A cat sitting on a mat."
    mock_client.models.generate_content.assert_called_once()
    call_kwargs = mock_client.models.generate_content.call_args
    assert call_kwargs.kwargs["model"] == "gemini-3.1-pro"
    contents = call_kwargs.kwargs["contents"]
    assert contents[1] == "What is in this image?"


def test_analyze_image_resolves_relative_path():
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = SimpleNamespace(text="result")

    with patch("tools.image_analyzer.genai") as mock_genai:
        mock_genai.Client.return_value = mock_client
        from tools.image_analyzer import create_image_analyzer

        tool = create_image_analyzer(model="gemini-3.1-pro", api_key="fake-key")

    with patch("builtins.open", mock_open(read_data=b"data")) as mocked_open:
        tool.invoke(
            {"file_path": "2023/validation/img.jpg", "question": "Describe this."}
        )

    mocked_open.assert_called_once_with(
        ".gaia-questions/files/2023/validation/img.jpg", "rb"
    )


def test_analyze_image_resolves_absolute_path():
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = SimpleNamespace(text="result")

    with patch("tools.image_analyzer.genai") as mock_genai:
        mock_genai.Client.return_value = mock_client
        from tools.image_analyzer import create_image_analyzer

        tool = create_image_analyzer(model="gemini-3.1-pro", api_key="fake-key")

    with patch("builtins.open", mock_open(read_data=b"data")) as mocked_open:
        tool.invoke(
            {"file_path": "/tmp/absolute/img.png", "question": "What is this?"}
        )

    mocked_open.assert_called_once_with("/tmp/absolute/img.png", "rb")


def test_analyze_image_detects_jpeg_mime_type():
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = SimpleNamespace(text="ok")

    with patch("tools.image_analyzer.genai") as mock_genai:
        mock_genai.Client.return_value = mock_client
        from tools.image_analyzer import create_image_analyzer

        tool = create_image_analyzer(model="gemini-3.1-pro", api_key="fake-key")

    with patch("builtins.open", mock_open(read_data=b"jpeg-data")):
        tool.invoke(
            {"file_path": "photo.jpeg", "question": "Describe."}
        )

    call_kwargs = mock_client.models.generate_content.call_args
    contents = call_kwargs.kwargs["contents"]
    image_part = contents[0]
    assert image_part.inline_data.mime_type == "image/jpeg"


def test_analyze_image_falls_back_to_octet_stream_for_unknown_type():
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = SimpleNamespace(text="ok")

    with patch("tools.image_analyzer.genai") as mock_genai:
        mock_genai.Client.return_value = mock_client
        from tools.image_analyzer import create_image_analyzer

        tool = create_image_analyzer(model="gemini-3.1-pro", api_key="fake-key")

    with patch("builtins.open", mock_open(read_data=b"unknown-data")):
        tool.invoke(
            {"file_path": "file.xyz123", "question": "What?"}
        )

    call_kwargs = mock_client.models.generate_content.call_args
    contents = call_kwargs.kwargs["contents"]
    image_part = contents[0]
    assert image_part.inline_data.mime_type == "application/octet-stream"


def test_analyze_image_passes_api_key_to_client():
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = SimpleNamespace(text="ok")

    with patch("tools.image_analyzer.genai") as mock_genai:
        mock_genai.Client.return_value = mock_client
        from tools.image_analyzer import create_image_analyzer

        create_image_analyzer(model="gemini-3.1-pro", api_key="my-secret-key")

    mock_genai.Client.assert_called_once_with(api_key="my-secret-key")
