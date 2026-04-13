from tools.tool_response import ToolError, ToolSuccess


def test_tool_success_default_fields():
    result = ToolSuccess()
    assert result.type == "success"
    assert result.response == ""


def test_tool_success_with_response():
    result = ToolSuccess(response="hello")
    assert result.type == "success"
    assert result.response == "hello"


def test_tool_error_default_fields():
    result = ToolError()
    assert result.type == "error"
    assert result.error == ""


def test_tool_error_with_message():
    result = ToolError(error="something broke")
    assert result.type == "error"
    assert result.error == "something broke"
