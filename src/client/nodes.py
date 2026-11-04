import json
import logging
import re

from openai import OpenAI

from src.client.core import DialogueState, McpClient, render_tool_result, system_message
from src.common.budget import LLM_MODEL, RESPONSE_RESERVE_TOKENS, TOKEN_COUNTER
from src.common.personas import Persona
from src.tools import LOOKUP_TOOLS
from src.tools.wikipedia import WikipediaTool

logger = logging.getLogger(__name__)


# ============
# Shared Nodes
# ============


class SharedNodes:
    """The nodes that are identical in both graphs, plus the dependencies every node class is bound to."""

    def __init__(self, mcp: McpClient, llm: OpenAI, persona: Persona, ambient: str):
        """Constructor."""
        self._mcp = mcp
        self._llm = llm
        self._persona = persona
        self._ambient = ambient

    # ==============
    # Public methods
    # ==============

    def input(self, state: DialogueState) -> DialogueState:
        """Receive the player's message and clear per-turn fields.

        Args:
            state: Turn state holding ``user_input``.

        Returns:
            The per-turn defaults.
        """
        logger.info("Player: %s", state["user_input"])
        return {"needs_tool": False, "tool_name": "", "tool_args": {}, "tool_calls": [], "tool_result": {}, "reply": ""}

    def check_context(self, state: DialogueState) -> DialogueState:
        """Ask the MCP server for the summarized context.

        Args:
            state: Turn state.

        Returns:
            ``summary`` and ``recent`` as the server returned them.
        """
        context = self._mcp.get_context()
        logger.info("Context: %s", context["tokens"])
        return {"summary": context["summary"], "recent": context["messages"]}

    def tool_call(self, state: DialogueState) -> DialogueState:
        """Run the chosen tool through the server.

        Args:
            state: Turn state holding ``tool_name`` and ``tool_args``.

        Returns:
            ``tool_result``.
        """
        result = self._mcp.tool_call(state["tool_name"], **state["tool_args"])
        logger.info("Tool result: %s", result.get("title") or result.get("error"))
        return {"tool_result": result}

    def output(self, state: DialogueState) -> DialogueState:
        """Log the reply and store the exchange on the server.

        Args:
            state: Turn state holding ``user_input`` and ``reply``.

        Returns:
            Nothing new.
        """
        logger.info("%s: %s", self._persona.name, state["reply"])
        self._mcp.add_message("user", state["user_input"])
        self._mcp.add_message("assistant", state["reply"])
        return {}


# ================
# Rule-Based Nodes
# ================


def is_factual_question(text: str) -> bool:
    """Decide whether a player message asks for general knowledge worth a tool lookup.

    Args:
        text: Player message.

    Returns:
        ``True`` when the message opens like a factual question (what, who, when, where, which, why, how many,
        tell me about, explain, ...) and does not refer to the NPC or the player (you, your, I, my, should, ...).
    """
    opens_factually = re.match(
        r"^\W*(what|who|whom|whose|when|where|which|why|how (many|much|does|do|did|is|are|was|were|long|far|old)"
        r"|tell me about|explain|define|describe)\b",
        text,
        re.IGNORECASE,
    )
    is_personal = re.search(r"\b(you|your|yourself|i|my|we|our|should)\b", text, re.IGNORECASE)
    return bool(opens_factually) and not is_personal


def extract_query(text: str) -> str:
    """Turn a factual question into a short search query.

    Args:
        text: Player message that passed :func:`is_factual_question`.

    Returns:
        The message without its leading question words and trailing punctuation.
    """
    query = re.sub(
        r"^\W*((what|who|when|where|which|why|how)(\s+(is|are|was|were|does|do|did|many|much))?"
        r"|tell me about|explain|define|describe)(\s+(the|a|an))?\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )
    return query.strip(" ?.!") or text


class RuleBasedNodes(SharedNodes):
    """Nodes of the rule-based graph: the client decides on the lookup before the prompt is built."""

    # ==============
    # Public methods
    # ==============

    def tool_decision(self, state: DialogueState) -> DialogueState:
        """Classify the message; the conditional edge reads ``needs_tool``.

        Args:
            state: Turn state holding ``user_input``.

        Returns:
            ``needs_tool``, the Wikipedia tool name and its ``query`` argument when the message is factual.
        """
        needs_tool = is_factual_question(state["user_input"])
        query = extract_query(state["user_input"]) if needs_tool else ""
        logger.info("Tool decision: %s%s", needs_tool, f" (query={query!r})" if needs_tool else "")
        return {"needs_tool": needs_tool, "tool_name": WikipediaTool.name, "tool_args": {"query": query} if needs_tool else {}}

    def prompt_assembly(self, state: DialogueState) -> DialogueState:
        """Build the GPT prompt with the tool result inside the system message, and measure it.

        Args:
            state: Turn state after the context and tool nodes have run.

        Returns:
            ``prompt`` (system message, recent turns verbatim, the new message) and ``prompt_tokens``.
        """
        prompt = [
            system_message(self._persona, self._ambient, state, render_tool_result(state)),
            *state.get("recent", []),
            {"role": "user", "content": state["user_input"]},
        ]
        return {"prompt": prompt, "prompt_tokens": TOKEN_COUNTER.count_messages(prompt)}

    def llm(self, state: DialogueState) -> DialogueState:
        """Send the prompt to GPT without any tools.

        Args:
            state: Turn state holding ``prompt``.

        Returns:
            ``reply``.
        """
        response = self._llm.chat.completions.create(
            model=LLM_MODEL, messages=state["prompt"], max_tokens=RESPONSE_RESERVE_TOKENS
        )
        return {"reply": response.choices[0].message.content.strip()}

    def route_after_decision(self, state: DialogueState) -> str:
        """Pick the next node from the tool decision.

        Args:
            state: Turn state holding ``needs_tool``.

        Returns:
            ``tool_call`` or ``prompt_assembly``.
        """
        return "tool_call" if state["needs_tool"] else "prompt_assembly"


# =============
# Agentic Nodes
# =============


def pick_tool_call(tool_calls: list[dict]) -> tuple[dict | None, str, dict]:
    """Find the first well-formed call to a lookup tool in a model answer.

    Args:
        tool_calls: Tool calls as stored in the state, each with ``id`` and ``function``.

    Returns:
        ``(call, name, args)`` for the first usable call, otherwise ``(None, "", {})``. A call is usable when it
        names a lookup tool, its arguments parse as a JSON object and every required argument is a non-empty string.
    """
    by_name = {tool.name: tool for tool in LOOKUP_TOOLS}
    for call in tool_calls:
        name = call["function"]["name"]
        tool = by_name.get(name)
        if tool is None:
            continue
        try:
            args = json.loads(call["function"]["arguments"])
        except json.JSONDecodeError:
            logger.warning("Unparseable tool arguments: %r", call["function"]["arguments"])
            continue
        if not isinstance(args, dict):
            continue
        required = tool.parameters.get("required", [])
        if all(isinstance(args.get(key), str) and args[key].strip() for key in required):
            return call, name, {key: value.strip() if isinstance(value, str) else value for key, value in args.items()}
    return None, "", {}


class AgenticNodes(SharedNodes):
    """Nodes of the agentic graph: GPT is offered the tool schemas and decides on the lookup itself."""

    # ==============
    # Public methods
    # ==============

    def prompt_assembly(self, state: DialogueState) -> DialogueState:
        """Build the GPT prompt, appending the tool exchange on the second pass, and measure it.

        Args:
            state: Turn state. ``tool_result`` is empty on the first pass of a turn.

        Returns:
            ``prompt`` (system message, recent turns, the new message and, after a lookup, the assistant's tool call
            followed by a ``tool`` message carrying the rendered result) and ``prompt_tokens``.
        """
        prompt = [
            system_message(self._persona, self._ambient, state),
            *state.get("recent", []),
            {"role": "user", "content": state["user_input"]},
        ]
        if state.get("tool_result"):
            call = state["tool_calls"][0]
            prompt += [
                {"role": "assistant", "content": None, "tool_calls": [call]},
                {"role": "tool", "tool_call_id": call["id"], "content": render_tool_result(state)},
            ]
        return {"prompt": prompt, "prompt_tokens": TOKEN_COUNTER.count_messages(prompt)}

    def llm(self, state: DialogueState) -> DialogueState:
        """Call GPT with the tool schemas attached and ``tool_choice="auto"`` on both passes.

        The schemas stay attached on the second pass because some providers reject a conversation that contains a
        tool exchange without them, and ``tool_choice="none"`` is not portable either. The loop is bounded by
        :meth:`tool_decision` instead, which ignores any tool call made after a lookup.

        Args:
            state: Turn state holding ``prompt``.

        Returns:
            ``reply`` (possibly empty) and the ``tool_calls`` GPT asked for.
        """
        response = self._llm.chat.completions.create(
            model=LLM_MODEL,
            messages=state["prompt"],
            tools=[tool.schema() for tool in LOOKUP_TOOLS],
            tool_choice="auto",
            max_tokens=RESPONSE_RESERVE_TOKENS,
        )
        message = response.choices[0].message
        calls = [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in message.tool_calls or []
        ]
        if calls and state.get("tool_result"):
            logger.warning("Model asked for another tool after the lookup; ignored, answering with its text.")
        return {"reply": (message.content or "").strip(), "tool_calls": calls}

    def tool_decision(self, state: DialogueState) -> DialogueState:
        """Read GPT's answer; the conditional edge reads ``needs_tool``. After a lookup the answer is final.

        Args:
            state: Turn state holding ``tool_calls``.

        Returns:
            ``needs_tool``, ``tool_name``, ``tool_args`` and the single kept ``tool_calls`` entry, or nothing on
            the second pass.
        """
        if state.get("tool_result"):
            return {}
        call, name, args = pick_tool_call(state.get("tool_calls", []))
        logger.info("Tool decision: %s%s", call is not None, f" ({name} {args})" if call else "")
        return {"needs_tool": call is not None, "tool_name": name, "tool_args": args, "tool_calls": [call] if call else []}

    def route_after_decision(self, state: DialogueState) -> str:
        """Run the tool once per turn, then finish.

        Args:
            state: Turn state holding ``needs_tool`` and ``tool_result``.

        Returns:
            ``tool_call`` or ``output``.
        """
        return "tool_call" if state["needs_tool"] and not state.get("tool_result") else "output"
