"""Document parsing tool powered by Docling.

Converts structured documents (PDFs, Excel, PowerPoint, images with text,
etc.) to Markdown using Docling's DocumentConverter.
"""

from docling.document_converter import DocumentConverter
from langchain_core.tools import tool

from agent_graph.file_paths import to_local_file_path

converter = DocumentConverter()


@tool
def parse_document(file_path: str) -> str:
    """Parse a document and return its content as Markdown.

    This tool is best suited for parsing documents where the structure and
    layout carries meaning. For example, PDFs, Excel files, PowerPoint files,
    and images containing text.

    Provide an agent-facing file path (for example the one listed under
    "Provided file path" in the system prompt). The tool resolves it to the
    real location on disk, converts the document, and returns its Markdown
    representation.
    """
    local_path = to_local_file_path(file_path)
    result = converter.convert(local_path)
    return result.document.export_to_markdown()
