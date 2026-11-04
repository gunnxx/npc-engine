from fastapi.testclient import TestClient
import httpx
from openai import AuthenticationError
import pytest

from src.common.personas import Persona
from src.ui.app import create_app


class FakeChat:
    """Stand-in for ``Chat`` that echoes the input, reports a tool when asked and records resets."""

    def __init__(self, mode: str = "rule-based"):
        """Constructor."""
        self.persona = Persona(id="adam", name="Adam", description="A sage.", system_prompt="You are Adam.")
        self.mode = mode
        self.sent: list[str] = []
        self.resets = 0

    def send(self, message: str) -> dict:
        """Return a canned final state, with a tool call in agentic mode."""
        self.sent.append(message)
        state = {"reply": f"Echo: {message}", "prompt_tokens": 42, "tool_name": "", "tool_args": {}}
        if self.mode == "agentic":
            state.update(tool_name="search_wikipedia", tool_args={"query": "dragons"})
        return state

    def reset(self):
        """Record the reset."""
        self.resets += 1


def test_page_and_info():
    """The page is served and the header data names the character and the mode."""
    http = TestClient(create_app(FakeChat(mode="agentic")))
    assert "<title>NPC Chat</title>" in http.get("/").text
    assert http.get("/api/info").json() == {"name": "Adam", "description": "A sage.", "mode": "agentic"}


def test_chat_and_reset_go_to_the_one_chat():
    """Messages are trimmed and answered by the chat; reset reaches it too."""
    chat = FakeChat()
    http = TestClient(create_app(chat))
    assert http.post("/api/chat", json={"message": "  hello  "}).json() == {"reply": "Echo: hello", "tool": None, "prompt_tokens": 42}
    assert http.post("/api/reset").json() == {"ok": True}
    assert chat.sent == ["hello"] and chat.resets == 1


def test_tool_is_reported():
    """A turn that ran a tool reports its name and arguments."""
    http = TestClient(create_app(FakeChat(mode="agentic")))
    assert http.post("/api/chat", json={"message": "dragons?"}).json()["tool"] == {"name": "search_wikipedia", "args": {"query": "dragons"}}


def test_empty_message_is_rejected():
    """Blank input is a 400, not a model call."""
    chat = FakeChat()
    http = TestClient(create_app(chat))
    assert http.post("/api/chat", json={"message": "   "}).status_code == 400 and chat.sent == []


def test_openai_failure_is_reported(monkeypatch: pytest.MonkeyPatch):
    """An OpenAI error becomes a 502 whose detail names the cause instead of a bare 500."""
    chat = FakeChat()
    http = TestClient(create_app(chat))

    def boom(message: str) -> dict:
        response = httpx.Response(401, request=httpx.Request("POST", "https://api.openai.com"))
        raise AuthenticationError("Incorrect API key", response=response, body=None)

    monkeypatch.setattr(chat, "send", boom)
    res = http.post("/api/chat", json={"message": "hello"})
    assert res.status_code == 502 and "Incorrect API key" in res.json()["detail"]
