import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from openai import OpenAIError
from pydantic import BaseModel
import requests

from src.client import Chat

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    """Body of ``POST /api/chat``."""

    message: str


def create_app(chat: Chat) -> FastAPI:
    """Build the FastAPI app that serves the page and drives one conversation.

    Args:
        chat: The conversation every request goes to.

    Returns:
        App serving ``GET /``, ``GET /api/info``, ``POST /api/chat`` and ``POST /api/reset``.
    """
    app = FastAPI(title="NPC Chat UI")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        """Serve the chat page."""
        return Path(__file__).with_name("index.html").read_text(encoding="utf-8")

    @app.get("/api/info")
    def info() -> dict:
        """Describe the character and the routing mode for the page header."""
        return {"name": chat.persona.name, "description": chat.persona.description, "mode": chat.mode}

    @app.post("/api/chat")
    def send(body: ChatRequest) -> dict:
        """Answer one player message."""
        if not body.message.strip():
            raise HTTPException(status_code=400, detail="Empty message.")
        try:
            state = chat.send(body.message.strip())
        except (requests.RequestException, RuntimeError) as exc:
            logger.warning("Turn failed: %s", exc)
            raise HTTPException(status_code=502, detail=f"MCP server error: {exc}") from exc
        except OpenAIError as exc:
            logger.warning("Turn failed: %s", exc)
            raise HTTPException(status_code=502, detail=f"OpenAI error: {exc}") from exc
        tool = {"name": state["tool_name"], "args": state.get("tool_args", {})} if state.get("tool_name") else None
        return {"reply": state["reply"], "tool": tool, "prompt_tokens": state.get("prompt_tokens", 0)}

    @app.post("/api/reset")
    def reset() -> dict:
        """Forget everything the character remembers."""
        try:
            chat.reset()
        except (requests.RequestException, RuntimeError) as exc:
            raise HTTPException(status_code=502, detail=f"MCP server error: {exc}") from exc
        return {"ok": True}

    return app
