from unittest.mock import MagicMock

from tools.image_analyzer import BaseImageAnalyzer, create_image_analyzer_tool


def _make_tool(analyzer: BaseImageAnalyzer):
    return create_image_analyzer_tool(analyzer=analyzer)


def test_analyze_image_delegates_to_analyzer():
    analyzer = MagicMock(spec=BaseImageAnalyzer)
    analyzer.answer_image_question.return_value = "A cat sitting on a mat."

    tool = _make_tool(analyzer)
    result = tool.invoke(
        {"file_path": "/tmp/cat.png", "question": "What is in this image?"}
    )

    assert result == "A cat sitting on a mat."
    analyzer.answer_image_question.assert_called_once_with(
        local_file_path="/tmp/cat.png", question="What is in this image?"
    )


def test_analyze_image_resolves_relative_path():
    analyzer = MagicMock(spec=BaseImageAnalyzer)
    analyzer.answer_image_question.return_value = "result"

    tool = _make_tool(analyzer)
    tool.invoke({"file_path": "2023/validation/img.jpg", "question": "Describe this."})

    analyzer.answer_image_question.assert_called_once_with(
        local_file_path=".gaia-questions/files/2023/validation/img.jpg",
        question="Describe this.",
    )


def test_analyze_image_preserves_absolute_path():
    analyzer = MagicMock(spec=BaseImageAnalyzer)
    analyzer.answer_image_question.return_value = "result"

    tool = _make_tool(analyzer)
    tool.invoke({"file_path": "/tmp/absolute/img.png", "question": "What is this?"})

    analyzer.answer_image_question.assert_called_once_with(
        local_file_path="/tmp/absolute/img.png", question="What is this?"
    )
