from fastapi.testclient import TestClient
import pytest

from src.server import app as mcp_app, memory as mcp_memory
from src.common.budget import RECENT_TOKENS, SUMMARY_TOKENS, TOKEN_COUNTER

client = TestClient(mcp_app.app)
"""HTTP client bound to the FastAPI app."""


def call(op: str, params: dict | None = None, session_id: str = "test") -> dict:
    """Post one envelope and return the parsed body.

    Args:
        op: Operation name.
        params: Operation parameters.
        session_id: Session to address.

    Returns:
        Response envelope as a dictionary.
    """
    return client.post("/mcp", json={"op": op, "session_id": session_id, "params": params or {}}).json()


def long_text(tokens: int, seed: str = "lore") -> str:
    """Build a text of the requested token count.

    Args:
        tokens: Target length.
        seed: Word to repeat.

    Returns:
        Text whose token count is exactly ``tokens``.
    """
    words = [seed] * tokens
    while TOKEN_COUNTER.count(" ".join(words)) > tokens:
        words.pop()
    text = " ".join(words)
    assert TOKEN_COUNTER.count(text) == tokens, "test helper assumes one token per word"
    return text


def test_add_and_get_context_roundtrip(fresh_memory: mcp_memory.ConversationMemory):
    """Messages come back in order with a token breakdown."""
    call("ADD_MESSAGE", {"role": "user", "content": "Hello Adam, I am Kael."})
    call("ADD_MESSAGE", {"role": "assistant", "content": "Welcome, Kael."})
    body = call("GET_CONTEXT")
    assert body["ok"]
    assert [m["role"] for m in body["result"]["messages"]] == ["user", "assistant"]
    assert body["result"]["summary"] == ""
    assert body["result"]["tokens"]["total"] == body["result"]["tokens"]["recent"] > 0


def test_sessions_are_isolated(fresh_memory: mcp_memory.ConversationMemory):
    """Two NPCs never see each other's history."""
    call("ADD_MESSAGE", {"role": "user", "content": "Secret for Adam."}, session_id="adam")
    assert call("GET_CONTEXT", session_id="mira")["result"]["messages"] == []


def test_reset_clears_session(fresh_memory: mcp_memory.ConversationMemory):
    """RESET empties both messages and summary."""
    call("ADD_MESSAGE", {"role": "user", "content": "Remember this."})
    call("SUMMARIZE_HISTORY")
    call("RESET")
    result = call("GET_CONTEXT")["result"]
    assert result["messages"] == [] and result["summary"] == ""


@pytest.mark.parametrize(
    ("op", "params", "fragment"),
    [
        ("NOPE", {}, "Unknown op"),
        ("ADD_MESSAGE", {"role": "user"}, "Missing parameter"),
        ("ADD_MESSAGE", {"role": "narrator", "content": "x"}, "role must be"),
        ("TOOL_CALL", {}, "Missing parameter"),
        ("TOOL_CALL", {"tool": "teleport"}, "Unknown tool"),
        ("TOOL_CALL", {"tool": "search_wikipedia", "args": {"topic": "x"}}, "Bad arguments"),
    ],
)
def test_errors_are_protocol_responses(fresh_memory: mcp_memory.ConversationMemory, op: str, params: dict, fragment: str):
    """Bad requests return ok=False with a message instead of an HTTP error."""
    body = call(op, params)
    assert body["ok"] is False and fragment in body["error"]


def test_tool_call_runs_named_tool(fresh_memory: mcp_memory.ConversationMemory):
    """TOOL_CALL dispatches to the registry with the given arguments."""
    result = call("TOOL_CALL", {"tool": "search_wikipedia", "args": {"query": "video game genres"}})["result"]
    assert result["title"] == "Video Game Genres" and "extract" in result
    clock = call("TOOL_CALL", {"tool": "clock_now"})["result"]
    assert clock["location"] == "Singapore" and set(clock) >= {"weekday", "time", "part_of_day", "is_weekend"}
    weather = call("TOOL_CALL", {"tool": "weather_now"})["result"]
    assert weather["condition"] == "an overcast sky"


def test_explicit_summarize_keeps_recent_messages(fresh_memory: mcp_memory.ConversationMemory):
    """SUMMARIZE_HISTORY folds all but the newest turns and reports the count."""
    for i in range(10):
        call("ADD_MESSAGE", {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"})
    body = call("SUMMARIZE_HISTORY")["result"]
    assert body["folded"] == 10 - 6
    assert body["summary"] == f"[+{10 - 6} msgs]"
    assert len(call("GET_CONTEXT")["result"]["messages"]) == 6


def test_auto_summarize_when_over_budget(fresh_memory: mcp_memory.ConversationMemory):
    """Adding messages past the recent-token budget triggers summarization automatically."""
    per_message = RECENT_TOKENS // 4
    for _ in range(6):
        call("ADD_MESSAGE", {"role": "user", "content": long_text(per_message)})
    result = call("GET_CONTEXT")["result"]
    assert result["summary"] == "[+1 msgs] [+1 msgs] [+1 msgs]"  # one fold per overflow, from the 4th message on
    assert len(result["messages"]) == 3
    assert result["tokens"]["recent"] <= RECENT_TOKENS
    assert result["tokens"]["recent"] <= RECENT_TOKENS
    assert result["tokens"]["total"] <= SUMMARY_TOKENS + RECENT_TOKENS


def test_low_importance_turns_dropped_before_summarizing(fresh_memory: mcp_memory.ConversationMemory):
    """Filler turns are dropped first; only then are the oldest meaningful turns folded into the summary."""
    big = long_text(RECENT_TOKENS // 2 - 50)
    for content in [big, "ok", "thanks", big, big]:
        call("ADD_MESSAGE", {"role": "user", "content": content})
    result = call("GET_CONTEXT")["result"]
    assert [m["content"] for m in result["messages"]] == [big, big]
    assert result["summary"] == "[+1 msgs]"
    assert result["tokens"]["recent"] <= RECENT_TOKENS


def test_score_importance_prefers_questions_and_numbers():
    """Filler scores zero; questions and numbers outrank plain statements of the same length."""
    assert mcp_memory.score_importance("ok") == 0
    assert mcp_memory.score_importance("Thanks!") == 0
    plain = mcp_memory.score_importance("I walked to the harbour")
    assert mcp_memory.score_importance("I walked to the harbour?") > plain
    assert mcp_memory.score_importance("I walked to harbour 7") > plain

