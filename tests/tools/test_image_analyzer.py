from unittest.mock import MagicMock

from tools.image_analyzer import BaseImageAnalyzer, create_image_analyzer_tool
from tools.tool_response import ToolError, ToolSuccess


def _make_tool(analyzer: BaseImageAnalyzer):
    return create_image_analyzer_tool(analyzer=analyzer)


def test_analyze_image_delegates_to_analyzer():
    analyzer = MagicMock(spec=BaseImageAnalyzer)
    analyzer.answer_image_question.return_value = "A cat sitting on a mat."

    tool = _make_tool(analyzer)
    result = tool.invoke(
        {"file_path": "/tmp/cat.png", "question": "What is in this image?"}
    )

    assert isinstance(result, ToolSuccess)
    assert result.response == "A cat sitting on a mat."
    analyzer.answer_image_question.assert_called_once_with(
        local_file_path="/tmp/cat.png", question="What is in this image?"
    )


def test_analyze_image_resolves_relative_path():
    analyzer = MagicMock(spec=BaseImageAnalyzer)
    analyzer.answer_image_question.return_value = "result"

    tool = _make_tool(analyzer)
    result = tool.invoke(
        {"file_path": "2023/validation/img.jpg", "question": "Describe this."}
    )

    assert isinstance(result, ToolSuccess)
    analyzer.answer_image_question.assert_called_once_with(
        local_file_path=".gaia-questions/files/2023/validation/img.jpg",
        question="Describe this.",
    )


def test_analyze_image_preserves_absolute_path():
    analyzer = MagicMock(spec=BaseImageAnalyzer)
    analyzer.answer_image_question.return_value = "result"

    tool = _make_tool(analyzer)
    result = tool.invoke(
        {"file_path": "/tmp/absolute/img.png", "question": "What is this?"}
    )

    assert isinstance(result, ToolSuccess)
    analyzer.answer_image_question.assert_called_once_with(
        local_file_path="/tmp/absolute/img.png", question="What is this?"
    )


def test_analyze_image_returns_tool_error_on_exception():
    analyzer = MagicMock(spec=BaseImageAnalyzer)
    analyzer.answer_image_question.side_effect = ConnectionError("API unavailable")

    tool = _make_tool(analyzer)
    result = tool.invoke(
        {"file_path": "/tmp/img.png", "question": "What is this?"}
    )

    assert isinstance(result, ToolError)
    assert "API unavailable" in result.error
