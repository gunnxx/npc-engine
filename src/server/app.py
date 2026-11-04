import logging

from fastapi import FastAPI
import requests

from src.common.budget import RECENT_TOKENS, SUMMARY_TOKENS, TOKEN_COUNTER
from src.common.protocol import McpRequest, McpResponse
from src.server.memory import ConversationMemory
from src.tools import TOOLS

logger = logging.getLogger(__name__)


# ===========
# Dispatcher
# ===========

memory = ConversationMemory()
"""Process-wide memory shared by every request."""


def handle_add_message(session_id: str, params: dict) -> dict:
    """Handle ``ADD_MESSAGE``: append one message (``role``, ``content``) to the session history.

    Args:
        session_id: Conversation identifier.
        params: ``role`` and ``content`` of the new turn.

    Returns:
        Message count and recent-token total after the append.
    """
    role, content = params["role"], params["content"]
    if role not in ("user", "assistant"):
        raise ValueError(f"role must be 'user' or 'assistant', got {role!r}.")
    session = memory.add_message(session_id, role, content)
    return {"message_count": len(session.messages), "recent_tokens": session.recent_tokens()}


def handle_get_context(session_id: str, params: dict) -> dict:
    """Handle ``GET_CONTEXT``: return the rolling summary plus the recent messages that fit the budget.

    Args:
        session_id: Conversation identifier.
        params: Unused.

    Returns:
        ``summary``, ``messages`` and a ``tokens`` breakdown that is guaranteed to fit the budget.
    """
    session = memory.get(session_id)
    summary_tokens = TOKEN_COUNTER.count(session.summary)
    return {
        "summary": session.summary,
        "messages": [message.to_dict() for message in session.messages],
        "tokens": {
            "summary": summary_tokens,
            "recent": session.recent_tokens(),
            "total": summary_tokens + session.recent_tokens(),
            "budget": SUMMARY_TOKENS + RECENT_TOKENS,
        },
    }


def handle_summarize_history(session_id: str, params: dict) -> dict:
    """Handle ``SUMMARIZE_HISTORY``: fold all but the most recent messages into the rolling summary.

    Args:
        session_id: Conversation identifier.
        params: Unused.

    Returns:
        The new ``summary`` and how many messages were ``folded`` into it.
    """
    folded = memory.summarize(session_id)
    return {"summary": memory.get(session_id).summary, "folded": folded}


def handle_reset(session_id: str, params: dict) -> dict:
    """Handle ``RESET``: clear the session history and summary.

    Args:
        session_id: Conversation identifier.
        params: Unused.

    Returns:
        The id of the cleared session.
    """
    memory.reset(session_id)
    return {"session_id": session_id}


def handle_tool_call(session_id: str, params: dict) -> dict:
    """Handle ``TOOL_CALL``: run the named tool with its arguments and return its result.

    Args:
        session_id: Conversation identifier (unused, tool results are not stored).
        params: ``tool`` name from :data:`src.tools.TOOLS` and an optional ``args`` dictionary.

    Returns:
        Whatever the tool returned; network failures come back as an ``error`` key rather than an exception.
    """
    name, args = params["tool"], params.get("args", {})
    tool = TOOLS.get(name)
    if tool is None:
        raise ValueError(f"Unknown tool {name!r}. Expected one of {sorted(TOOLS)}.")
    try:
        return tool.run(**args)
    except TypeError as exc:
        raise ValueError(f"Bad arguments for {name}: {exc}") from exc
    except requests.RequestException as exc:
        logger.warning("Tool %s failed: %s", name, exc)
        return {"error": f"{name} is unavailable right now: {exc}"}


HANDLERS = {
    "ADD_MESSAGE": handle_add_message,
    "GET_CONTEXT": handle_get_context,
    "SUMMARIZE_HISTORY": handle_summarize_history,
    "RESET": handle_reset,
    "TOOL_CALL": handle_tool_call,
}
"""Operation name to handler function; the keys are the protocol's operation names, in assessment order."""


def dispatch(request: McpRequest) -> McpResponse:
    """Route a request envelope to its handler and wrap the outcome.

    Args:
        request: Parsed request envelope.

    Returns:
        Response envelope; failures are reported as ``ok=False`` instead of raising.
    """
    handler = HANDLERS.get(request.op)
    if handler is None:
        return McpResponse(ok=False, error=f"Unknown op {request.op!r}. Expected one of {sorted(HANDLERS)}.")
    try:
        return McpResponse(ok=True, result=handler(request.session_id, request.params))
    except KeyError as exc:
        return McpResponse(ok=False, error=f"Missing parameter {exc} for {request.op}.")
    except ValueError as exc:
        return McpResponse(ok=False, error=str(exc))
    except Exception as exc:  # noqa: BLE001 - every failure must come back as a protocol error
        logger.exception("Operation %s failed.", request.op)
        return McpResponse(ok=False, error=f"{type(exc).__name__}: {exc}")


# ====
# App
# ====

app = FastAPI(title="NPC MCP Server")


@app.post("/mcp", response_model=McpResponse)
def mcp_endpoint(request: McpRequest) -> McpResponse:
    """Accept one MCP request envelope.

    Args:
        request: Parsed request envelope.

    Returns:
        Response envelope from :func:`dispatch`.
    """
    logger.info("%s session=%s params=%s", request.op, request.session_id, request.params)
    return dispatch(request)


@app.get("/health")
def health() -> dict:
    """Liveness probe.

    Returns:
        ``{"status": "ok"}``.
    """
    return {"status": "ok"}

