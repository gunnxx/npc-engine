from pydantic import BaseModel, Field


class McpRequest(BaseModel):
    """Request envelope posted to ``/mcp``; every answer is an :class:`McpResponse`."""

    op: str
    """Operation name, a key of ``HANDLERS`` in ``src/server/app.py``."""

    session_id: str
    """Conversation the operation applies to. One session per NPC keeps their memories apart."""

    params: dict = Field(default_factory=dict)
    """Operation-specific parameters, see the handler docstrings in ``src/server/app.py``."""


class McpResponse(BaseModel):
    """Response envelope returned by the MCP server."""

    ok: bool
    """``True`` when the operation succeeded and ``result`` is set."""

    result: dict | None = None
    """Operation-specific payload, present when ``ok`` is ``True``."""

    error: str | None = None
    """Human-readable error, present when ``ok`` is ``False``."""
