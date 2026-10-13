from ddgs import DDGS
from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web. Returns titles, URLs and snippets for the top results."""
    results = DDGS().text(query, max_results=5)
    return "\n\n".join(f"{r['title']}\n{r['href']}\n{r['body']}" for r in results) or "No results."
