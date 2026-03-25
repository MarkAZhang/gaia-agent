from langchain_tavily import TavilySearch


def create_web_search(max_results: int = 5) -> TavilySearch:
    return TavilySearch(max_results=max_results)
