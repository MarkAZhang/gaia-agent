from unittest.mock import MagicMock, patch


@patch("tools.web_searcher.TavilySearch")
def test_create_web_search_default_max_results(mock_tavily_cls):
    from tools.web_searcher import create_web_search

    tool = create_web_search()

    mock_tavily_cls.assert_called_once_with(max_results=5)
    assert tool == mock_tavily_cls.return_value


@patch("tools.web_searcher.TavilySearch")
def test_create_web_search_custom_max_results(mock_tavily_cls):
    from tools.web_searcher import create_web_search

    create_web_search(max_results=10)

    mock_tavily_cls.assert_called_once_with(max_results=10)


@patch("tools.web_searcher.TavilySearch")
def test_web_search_tool_is_invocable(mock_tavily_cls):
    mock_instance = MagicMock()

    # Note: We don't return the actual
    mock_instance.invoke.return_value = {
        "query": "LangGraph framework",
        "results": [
            {
                "title": "LangGraph: A Framework for Building LLM Agents",
                "url": "https://example.com",
                "content": "LangGraph is a framework for building LLM agents.",
                "score": 0.95,
            },
        ],
        "images": [],
        "response_time": 0.5,
    }
    mock_tavily_cls.return_value = mock_instance

    from tools.web_searcher import create_web_search

    tool = create_web_search()
    result = tool.invoke({"query": "LangGraph framework"})

    mock_instance.invoke.assert_called_once_with({"query": "LangGraph framework"})
    assert len(result["results"]) == 1
    assert result["results"][0]["url"] == "https://example.com"


@patch("tools.web_searcher.TavilySearch")
def test_web_search_returns_empty_for_no_results(mock_tavily_cls):
    mock_instance = MagicMock()
    mock_instance.invoke.return_value = []
    mock_tavily_cls.return_value = mock_instance

    from tools.web_searcher import create_web_search

    tool = create_web_search()
    result = tool.invoke({"query": "xyznonexistentquery"})

    assert result == []
