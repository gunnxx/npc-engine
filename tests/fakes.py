import json
from types import SimpleNamespace


class FakeMcp:
    """Stand-in for ``McpClient`` that records calls."""

    def __init__(self, summary: str = "", recent: list[dict] | None = None):
        """Constructor."""
        self.summary = summary
        self.recent = recent or []
        self.added: list[tuple[str, str]] = []
        self.calls: list[tuple[str, dict]] = []

    def get_context(self) -> dict:
        """Return the canned context."""
        return {"summary": self.summary, "messages": self.recent, "tokens": {}}

    def tool_call(self, name: str, **args: object) -> dict:
        """Record the call and return a canned result for the tool."""
        self.calls.append((name, args))
        if name == "search_wikipedia":
            return {"title": "Video game genre", "extract": "Genres classify games by gameplay.", "url": "u"}
        if name == "clock_now":
            return {"location": "Singapore", "iso": "2026-11-03T23:40+08:00", "weekday": "Tuesday", "time": "23:40",
                    "part_of_day": "night", "is_weekend": False}
        if name == "weather_now":
            return {"error": "weather_now is unavailable right now: timeout"}
        raise RuntimeError(f"MCP TOOL_CALL failed: Unknown tool {name!r}.")

    def add_message(self, role: str, content: str) -> dict:
        """Record the stored turn."""
        self.added.append((role, content))
        return {}


def tool_call(query: object = "video game genres", name: str = "search_wikipedia", raw: str | None = None) -> SimpleNamespace:
    """Build one entry of ``message.tool_calls`` as the OpenAI SDK shapes it.

    Args:
        query: Value of the ``query`` argument.
        name: Function name the fake model chose.
        raw: Raw argument string, overriding ``query`` to simulate malformed JSON.

    Returns:
        Object with ``id`` and a ``function`` attribute carrying ``name`` and ``arguments``.
    """
    arguments = raw if raw is not None else json.dumps({"query": query})
    return SimpleNamespace(id="call_1", function=SimpleNamespace(name=name, arguments=arguments))


def fake_llm(reply: str = "Ah, traveller.", tool_calls: list[SimpleNamespace] | None = None) -> SimpleNamespace:
    """Build an object with the ``chat.completions.create`` shape.

    A request that offers tools and carries no ``tool`` message yet is answered with ``tool_calls`` and no text
    when ``tool_calls`` is given; every other request is answered with ``reply``.

    Args:
        reply: Text the fake model answers with.
        tool_calls: What the fake model answers a tool-enabled request with, ``None`` for a plain text answer.

    Returns:
        Object usable in place of an ``OpenAI`` client, with the recorded requests in ``calls``.
    """
    calls: list[dict] = []

    def create(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        first_pass = kwargs.get("tools") and not any(m.get("role") == "tool" for m in kwargs["messages"])
        if tool_calls and first_pass:
            message = SimpleNamespace(content=None, tool_calls=tool_calls)
        else:
            message = SimpleNamespace(content=reply, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), calls=calls)
