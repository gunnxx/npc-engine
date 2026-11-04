import pytest

from src.server import app as mcp_app, memory as mcp_memory
from src.tools import TOOLS


@pytest.fixture
def fresh_memory(monkeypatch: pytest.MonkeyPatch) -> mcp_memory.ConversationMemory:
    """Give each test an empty memory and fake the three external services (GPT, Wikipedia, Open-Meteo).

    Args:
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        The memory instance the server handlers will use during the test.
    """
    memory = mcp_memory.ConversationMemory()
    monkeypatch.setattr(mcp_app, "memory", memory)
    monkeypatch.setattr(
        mcp_memory,
        "summarize_with_llm",
        lambda previous, messages: f"{previous} [+{len(messages)} msgs]".strip(),
    )
    monkeypatch.setattr(
        TOOLS["search_wikipedia"],
        "run",
        lambda query: {"title": query.title(), "extract": f"Facts about {query}.", "url": "https://example.org"},
    )
    monkeypatch.setattr(
        TOOLS["weather_now"],
        "run",
        lambda: {"location": "Singapore", "temperature_c": 27.2, "humidity_pct": 91, "condition": "an overcast sky",
                 "is_day": False, "observed_at": "2026-11-03T18:15"},
    )
    return memory
