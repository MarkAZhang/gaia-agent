from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def test_parse_document_converts_and_returns_markdown():
    mock_document = MagicMock()
    mock_document.export_to_markdown.return_value = "# Title\n\nSome content"
    mock_result = SimpleNamespace(document=mock_document)

    with patch("tools.document_parser.converter") as mock_converter:
        mock_converter.convert.return_value = mock_result
        from tools.document_parser import parse_document

        result = parse_document.invoke({"file_path": "2023/validation/doc.pdf"})

    mock_converter.convert.assert_called_once_with(
        ".gaia-questions/files/2023/validation/doc.pdf"
    )
    assert result == "# Title\n\nSome content"


def test_parse_document_resolves_absolute_path():
    mock_document = MagicMock()
    mock_document.export_to_markdown.return_value = "content"
    mock_result = SimpleNamespace(document=mock_document)

    with patch("tools.document_parser.converter") as mock_converter:
        mock_converter.convert.return_value = mock_result
        from tools.document_parser import parse_document

        parse_document.invoke({"file_path": "/tmp/absolute/doc.xlsx"})

    mock_converter.convert.assert_called_once_with("/tmp/absolute/doc.xlsx")


def test_parse_document_resolves_already_rooted_path():
    mock_document = MagicMock()
    mock_document.export_to_markdown.return_value = "md"
    mock_result = SimpleNamespace(document=mock_document)

    with patch("tools.document_parser.converter") as mock_converter:
        mock_converter.convert.return_value = mock_result
        from tools.document_parser import parse_document

        parse_document.invoke(
            {"file_path": ".gaia-questions/files/2023/validation/doc.pptx"}
        )

    mock_converter.convert.assert_called_once_with(
        ".gaia-questions/files/2023/validation/doc.pptx"
    )


def test_parse_document_returns_empty_string_for_empty_document():
    mock_document = MagicMock()
    mock_document.export_to_markdown.return_value = ""
    mock_result = SimpleNamespace(document=mock_document)

    with patch("tools.document_parser.converter") as mock_converter:
        mock_converter.convert.return_value = mock_result
        from tools.document_parser import parse_document

        result = parse_document.invoke({"file_path": "empty.pdf"})

    assert result == ""
