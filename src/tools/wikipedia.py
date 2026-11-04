import requests

from src.common.budget import TOKEN_COUNTER, TOOL_RESULT_TOKENS
from src.tools.base import HTTP_TIMEOUT_SECONDS, Tool


class WikipediaTool(Tool):
    """Search Wikipedia and return the top article's lead section."""

    name = "search_wikipedia"
    """Tool identifier."""

    description = (
        "Look up a general-knowledge topic in an encyclopedia: history, geography, science, culture, games, "
        "real people and real places. Use it when the player asks about the real world and a wrong answer "
        "would matter. Do not use it for small talk, for questions about yourself, or for the player's own "
        "situation, feelings and choices."
    )
    """Routing policy GPT reads in the agentic graph."""

    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Short search query naming the topic, for example 'Leif Erikson'."}
        },
        "required": ["query"],
    }
    """One required string argument."""

    # ==============
    # Public methods
    # ==============

    def run(self, query: str) -> dict:
        """Return the lead section of the Wikipedia page that best matches a query.

        Args:
            query: Free-text search query.

        Returns:
            ``title``, ``extract`` and ``url``, or an ``error`` key when nothing was found. The MediaWiki search API
            picks the page and the REST summary API returns its lead section; Wikipedia requires a descriptive
            User-Agent on both.
        """
        headers = {"User-Agent": "npc-engine/0.1 (technical assessment; contact via repository)"}
        search = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": query, "srlimit": 1, "format": "json"},
            headers=headers,
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        search.raise_for_status()
        hits = search.json()["query"]["search"]
        if not hits:
            return {"error": f"No Wikipedia article found for {query!r}."}
        title = hits[0]["title"]
        page = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}",
            headers=headers,
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        page.raise_for_status()
        data = page.json()
        return {
            "title": data.get("title", title),
            "extract": TOKEN_COUNTER.truncate(data.get("extract", ""), TOOL_RESULT_TOKENS),
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
        }

    def render(self, result: dict) -> str:
        """Present the article as a Lorebook entry, or admit the Lorebook has nothing.

        Args:
            result: What :meth:`run` returned.

        Returns:
            Text for the model.
        """
        if result.get("extract"):
            return f"Lorebook entry on \"{result['title']}\":\n{result['extract']}"
        return "The Lorebook has no entry for this question; admit what you do not know."
