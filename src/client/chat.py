from openai import OpenAI

from src.client import agentic_graph, rule_based_graph
from src.client.core import DialogueState, McpClient, gather_ambient
from src.common.personas import Persona

GRAPH_BUILDERS = {"rule-based": rule_based_graph.build_graph, "agentic": agentic_graph.build_graph}
"""Graph factory for every routing mode."""

DEFAULT_MODE = "rule-based"
"""Routing mode used when none is chosen on the command line."""


class Chat:
    """One conversation: a character, a routing mode and the compiled graph that answers for them.

    Both front ends run exactly one. Building it connects to the character's session on the MCP server, gathers the
    ambient note once and compiles the graph.
    """

    def __init__(self, persona: Persona, mode: str, llm: OpenAI, server_url: str):
        """Constructor."""
        self.persona = persona
        self.mode = mode
        self.mcp = McpClient(server_url, session_id=persona.id)
        self.ambient = gather_ambient(self.mcp)
        self._graph = GRAPH_BUILDERS[mode](self.mcp, llm, persona, self.ambient)

    # ==============
    # Public methods
    # ==============

    def send(self, message: str) -> DialogueState:
        """Run one turn.

        Args:
            message: The player's message.

        Returns:
            The final turn state, with ``reply``, ``tool_name``, ``tool_args``, ``tool_result`` and ``prompt_tokens``.
        """
        return self._graph.invoke({"user_input": message})

    def reset(self):
        """Forget everything the character remembers."""
        self.mcp.reset()
