import logging
from typing import TypedDict

import requests

from src.common.budget import AMBIENT_TOKENS, TOKEN_COUNTER
from src.common.personas import Persona
from src.tools import TOOLS

logger = logging.getLogger(__name__)


# ==========
# MCP Client
# ==========


class McpClient:
    """Thin HTTP wrapper around the ``POST /mcp`` envelope."""

    def __init__(self, base_url: str, session_id: str):
        """Constructor."""
        self._url = f"{base_url.rstrip('/')}/mcp"
        self._session_id = session_id
        self._http = requests.Session()

    # ==============
    # Public methods
    # ==============

    def add_message(self, role: str, content: str) -> dict:
        """Append a turn to the NPC's memory.

        Args:
            role: ``user`` or ``assistant``.
            content: Text of the turn.

        Returns:
            Server result with ``message_count`` and ``recent_tokens``.
        """
        return self._call("ADD_MESSAGE", {"role": role, "content": content})

    def get_context(self) -> dict:
        """Fetch the summary and recent messages that fit the budget.

        Returns:
            Server result with ``summary``, ``messages`` and ``tokens``.
        """
        return self._call("GET_CONTEXT")

    def tool_call(self, name: str, **args: object) -> dict:
        """Run a server-side tool.

        Args:
            name: Tool name, a key of :data:`src.tools.TOOLS`.
            **args: Arguments matching the tool's parameter schema.

        Returns:
            The tool's result, which carries an ``error`` key when it found nothing or was unavailable.
        """
        return self._call("TOOL_CALL", {"tool": name, "args": args})

    def summarize_history(self) -> dict:
        """Fold the older turns into the rolling summary now, without waiting for the budget to force it.

        Returns:
            Server result with the new ``summary`` and the number of messages ``folded``.
        """
        return self._call("SUMMARIZE_HISTORY")

    def reset(self) -> dict:
        """Clear the NPC's memory.

        Returns:
            Server result naming the cleared session.
        """
        return self._call("RESET")

    # ===============
    # Private methods
    # ===============

    def _call(self, op: str, params: dict | None = None) -> dict:
        """Send one envelope and unwrap the response.

        Args:
            op: Operation name.
            params: Operation parameters.

        Returns:
            The ``result`` payload.
        """
        response = self._http.post(self._url, json={"op": op, "session_id": self._session_id, "params": params or {}})
        response.raise_for_status()
        body = response.json()
        if not body["ok"]:
            raise RuntimeError(f"MCP {op} failed: {body['error']}")
        return body["result"]


# ==========
# Turn State
# ==========


class DialogueState(TypedDict, total=False):
    """State passed between the nodes of one dialogue turn; both graphs use the same fields."""

    user_input: str
    """The player's new message."""

    summary: str
    """Rolling summary returned by the server."""

    recent: list[dict]
    """Recent messages returned by the server."""

    needs_tool: bool
    """Whether this turn runs a lookup tool."""

    tool_name: str
    """Tool to run when ``needs_tool`` is set, a key of :data:`src.tools.TOOLS`."""

    tool_args: dict
    """Arguments passed to that tool."""

    tool_calls: list[dict]
    """Tool calls GPT asked for, in OpenAI's message format. Only the agentic graph fills this."""

    tool_result: dict
    """Tool payload, empty when no tool ran."""

    prompt: list[dict]
    """Full chat prompt sent to GPT."""

    prompt_tokens: int
    """Measured size of ``prompt``."""

    reply: str
    """The NPC's answer."""


# ==============
# Prompt Helpers
# ==============


def system_message(persona: Persona, ambient: str, state: DialogueState, extra: str = "") -> dict:
    """Build the system message from the persona, the memory summary, the ambient note and an optional extra block.

    Args:
        persona: Character the graph speaks as.
        ambient: Note from :func:`gather_ambient`, empty when no ambient tool answered.
        state: Turn state after the context node has run.
        extra: Text appended as a final paragraph, for example a Lorebook entry.

    Returns:
        One chat message with role ``system``.
    """
    parts = [persona.system_prompt]
    if state.get("summary"):
        parts.append(f"What you remember from earlier in this conversation:\n{state['summary']}")
    if ambient:
        parts.append(ambient)
    if extra:
        parts.append(extra)
    return {"role": "system", "content": "\n\n".join(parts)}


def render_tool_result(state: DialogueState) -> str:
    """Render the turn's tool result as the in-character text GPT reads.

    Args:
        state: Turn state after the tool node has run.

    Returns:
        The producing tool's rendering, or an empty string when no tool ran this turn.
    """
    if not state.get("tool_result"):
        return ""
    return TOOLS[state["tool_name"]].render(state["tool_result"])


def gather_ambient(mcp: McpClient) -> str:
    """Run every ambient tool once and compose the note about the player's real-world setting.

    A tool that fails or is unavailable is skipped with a warning, so a network problem never blocks the chat.

    Args:
        mcp: Client bound to the character's session.

    Returns:
        The preamble followed by one sentence per tool, cut to :data:`src.common.budget.AMBIENT_TOKENS`, or an
        empty string when no tool answered.
    """
    sentences = []
    for tool in (tool for tool in TOOLS.values() if tool.ambient):
        try:
            result = mcp.tool_call(tool.name)
        except (RuntimeError, requests.RequestException) as exc:
            logger.warning("Ambient tool %s skipped: %s", tool.name, exc)
            continue
        if "error" in result:
            logger.warning("Ambient tool %s skipped: %s", tool.name, result["error"])
            continue
        sentences.append(tool.render(result))
    if not sentences:
        return ""
    preamble = (
        "The player's real-world setting right now, for a light touch of small talk when it fits naturally. Never "
        "recite it and never mention how you know it. "
    )
    note = TOKEN_COUNTER.truncate(preamble + " ".join(sentences), AMBIENT_TOKENS)
    logger.info("Ambient: %s", note)
    return note
